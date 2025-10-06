#!/usr/bin/env python3
"""
SQLGenie MCP Server
A Model Context Protocol server that provides database query capabilities to Claude Desktop.
"""

import asyncio
import json
import sqlite3
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

# Directory structure for database storage (same as main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADED_DBS_DIR = os.path.join(BASE_DIR, 'data', 'uploaded_databases')
SCHEMA_CACHE_DIR = os.path.join(BASE_DIR, 'data', 'schema_cache')
DEFAULT_DB_PATH = os.path.join(BASE_DIR, 'data', 'sakila.db')

# Create directories if they don't exist
os.makedirs(UPLOADED_DBS_DIR, exist_ok=True)
os.makedirs(SCHEMA_CACHE_DIR, exist_ok=True)

class DatabaseManager:
    """Manages database operations for the MCP server"""
    
    def __init__(self):
        self.databases = {}
        self._load_databases()
    
    def _load_databases(self):
        """Load information about available databases"""
        self.databases = {}
        
        # Add default Sakila database
        if os.path.exists(DEFAULT_DB_PATH):
            self.databases['sakila'] = {
                'id': 'sakila',
                'name': 'Sakila Sample Database',
                'path': DEFAULT_DB_PATH,
                'tables': self._get_tables(DEFAULT_DB_PATH),
                'upload_date': 'Built-in',
                'file_size': os.path.getsize(DEFAULT_DB_PATH)
            }
        
        # Load uploaded databases
        if os.path.exists(UPLOADED_DBS_DIR):
            for db_file in os.listdir(UPLOADED_DBS_DIR):
                if db_file.endswith('.db'):
                    db_path = os.path.join(UPLOADED_DBS_DIR, db_file)
                    db_id = db_file.replace('.db', '')
                    
                    # Try to get original name from schema cache
                    schema_file = os.path.join(SCHEMA_CACHE_DIR, f"{db_id}.json")
                    original_name = db_file
                    
                    if os.path.exists(schema_file):
                        try:
                            with open(schema_file, 'r') as f:
                                schema_data = json.load(f)
                                original_name = schema_data.get('original_name', db_file)
                        except:
                            pass
                    
                    self.databases[db_id] = {
                        'id': db_id,
                        'name': original_name,
                        'path': db_path,
                        'tables': self._get_tables(db_path),
                        'upload_date': datetime.fromtimestamp(os.path.getctime(db_path)).isoformat(),
                        'file_size': os.path.getsize(db_path)
                    }
    
    def _get_tables(self, db_path: str) -> List[str]:
        """Get list of tables in a database"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            return tables
        except Exception:
            return []
    
    def get_database_list(self) -> List[Dict]:
        """Get list of all available databases"""
        self._load_databases()  # Refresh the list
        return list(self.databases.values())
    
    def get_database_schema(self, database_id: str) -> Dict:
        """Get detailed schema information for a database"""
        if database_id not in self.databases:
            raise ValueError(f"Database '{database_id}' not found")
        
        db_info = self.databases[database_id]
        db_path = db_info['path']
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            schema = {
                'database_id': database_id,
                'database_name': db_info['name'],
                'tables': {}
            }
            
            # Get tables and their columns
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                # Get column information
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                schema['tables'][table_name] = {
                    'columns': [
                        {
                            'name': col[1],
                            'type': col[2],
                            'not_null': bool(col[3]),
                            'default_value': col[4],
                            'primary_key': bool(col[5])
                        }
                        for col in columns
                    ]
                }
                
                # Get foreign keys
                cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                fks = cursor.fetchall()
                if fks:
                    schema['tables'][table_name]['foreign_keys'] = [
                        {
                            'column': fk[3],
                            'references_table': fk[2],
                            'references_column': fk[4]
                        }
                        for fk in fks
                    ]
            
            conn.close()
            return schema
            
        except Exception as e:
            raise ValueError(f"Error reading schema: {str(e)}")
    
    def execute_query(self, database_id: str, query: str) -> Dict:
        """Execute a SQL query on the specified database with safety checks"""
        if database_id not in self.databases:
            raise ValueError(f"Database '{database_id}' not found")
        
        # Basic safety checks
        query_upper = query.strip().upper()
        
        # Block potentially dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        if any(keyword in query_upper for keyword in dangerous_keywords):
            # Only allow these operations if explicitly confirmed
            if not query_upper.startswith(('SELECT', 'WITH', 'PRAGMA', 'EXPLAIN')):
                return {
                    'success': False,
                    'query': query,
                    'error': 'Potentially dangerous operation detected. This MCP server only allows read operations for safety.',
                    'error_type': 'SecurityError',
                    'allowed_operations': ['SELECT', 'WITH', 'PRAGMA', 'EXPLAIN']
                }
        
        # Limit query length
        if len(query) > 10000:
            return {
                'success': False,
                'query': query[:100] + "...",
                'error': 'Query too long. Maximum length is 10,000 characters.',
                'error_type': 'ValidationError'
            }
        
        db_path = self.databases[database_id]['path']
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            
            # Set a timeout for query execution
            conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds
            
            # Execute the query
            cursor.execute(query)
            
            # Handle different types of queries
            if query.strip().upper().startswith(('SELECT', 'WITH', 'PRAGMA', 'EXPLAIN')):
                # For SELECT queries, fetch results with a reasonable limit
                rows = cursor.fetchmany(1000)  # Limit to 1000 rows for safety
                columns = [description[0] for description in cursor.description] if cursor.description else []
                
                # Convert rows to list of dictionaries
                results = []
                for row in rows:
                    results.append({col: row[col] for col in columns})
                
                # Check if there are more rows
                has_more = len(cursor.fetchmany(1)) > 0
                
                response = {
                    'success': True,
                    'query': query,
                    'columns': columns,
                    'results': results,
                    'row_count': len(results),
                    'has_more_rows': has_more,
                    'query_type': 'SELECT'
                }
                
                if has_more:
                    response['warning'] = 'Results limited to first 1000 rows'
                    
            else:
                # This shouldn't happen due to our safety checks above
                response = {
                    'success': False,
                    'query': query,
                    'error': 'Only read operations are allowed',
                    'error_type': 'SecurityError'
                }
            
            conn.close()
            return response
            
        except sqlite3.OperationalError as e:
            return {
                'success': False,
                'query': query,
                'error': f"SQL Error: {str(e)}",
                'error_type': 'SQLError',
                'hint': 'Check your SQL syntax and table/column names'
            }
        except sqlite3.Error as e:
            return {
                'success': False,
                'query': query,
                'error': f"Database Error: {str(e)}",
                'error_type': 'DatabaseError'
            }
        except Exception as e:
            return {
                'success': False,
                'query': query,
                'error': f"Unexpected Error: {str(e)}",
                'error_type': type(e).__name__
            }

# Initialize the database manager
db_manager = DatabaseManager()

# Create the MCP server
server = Server("sqlgenie-mcp-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available MCP tools"""
    return [
        types.Tool(
            name="list_databases",
            description="List all available databases with their information",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get_database_schema",
            description="Get detailed schema information for a specific database including tables, columns, types, and relationships",
            inputSchema={
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "string",
                        "description": "The ID of the database to get schema for"
                    }
                },
                "required": ["database_id"]
            }
        ),
        types.Tool(
            name="execute_sql_query",
            description="Execute a SQL query on a specific database and return results",
            inputSchema={
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "string",
                        "description": "The ID of the database to query"
                    },
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute"
                    }
                },
                "required": ["database_id", "query"]
            }
        ),
        types.Tool(
            name="analyze_query_results",
            description="Get a natural language explanation of query results and insights",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query that was executed"
                    },
                    "results": {
                        "type": "array",
                        "description": "The query results to analyze"
                    },
                    "database_id": {
                        "type": "string",
                        "description": "The database ID for context"
                    }
                },
                "required": ["query", "results", "database_id"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "list_databases":
        try:
            databases = db_manager.get_database_list()
            result = {
                "databases": databases,
                "total_count": len(databases)
            }
            return [types.TextContent(
                type="text",
                text=f"Available Databases:\n\n{json.dumps(result, indent=2)}"
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error listing databases: {str(e)}"
            )]
    
    elif name == "get_database_schema":
        try:
            database_id = arguments.get("database_id")
            if not database_id:
                return [types.TextContent(
                    type="text",
                    text="Error: database_id is required"
                )]
            
            schema = db_manager.get_database_schema(database_id)
            return [types.TextContent(
                type="text",
                text=f"Database Schema for '{database_id}':\n\n{json.dumps(schema, indent=2)}"
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error getting schema: {str(e)}"
            )]
    
    elif name == "execute_sql_query":
        try:
            database_id = arguments.get("database_id")
            query = arguments.get("query")
            
            if not database_id or not query:
                return [types.TextContent(
                    type="text",
                    text="Error: Both database_id and query are required"
                )]
            
            result = db_manager.execute_query(database_id, query)
            return [types.TextContent(
                type="text",
                text=f"Query Execution Result:\n\n{json.dumps(result, indent=2, default=str)}"
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error executing query: {str(e)}"
            )]
    
    elif name == "analyze_query_results":
        try:
            query = arguments.get("query", "")
            results = arguments.get("results", [])
            database_id = arguments.get("database_id", "")
            
            # Create a summary analysis
            analysis = {
                "query_summary": f"Executed query: {query}",
                "result_count": len(results),
                "database": database_id
            }
            
            if results:
                analysis["sample_data"] = results[:5]  # First 5 rows as sample
                if len(results) > 5:
                    analysis["note"] = f"Showing first 5 rows of {len(results)} total results"
            
            return [types.TextContent(
                type="text",
                text=f"Query Analysis:\n\n{json.dumps(analysis, indent=2, default=str)}"
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error analyzing results: {str(e)}"
            )]
    
    else:
        return [types.TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]

async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="sqlgenie-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())