# SQLGenie MCP Integration Guide

## Overview
This guide explains how to integrate SQLGenie with Claude Desktop using the Model Context Protocol (MCP). This integration allows Claude Desktop to directly interact with your databases through MCP tools - completely free!

## What is MCP?
Model Context Protocol (MCP) is an open standard that enables AI assistants to securely connect to external data sources and tools. With this integration, Claude Desktop can:
- List your available databases
- Explore database schemas
- Execute SQL queries
- Analyze query results

## Prerequisites
1. **Claude Desktop** - Download from [claude.ai](https://claude.ai/download)
2. **Python 3.8+** with pip
3. **SQLGenie project** (this repository)

## Installation Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Claude Desktop
1. **Find Claude Desktop config location:**
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

2. **Add the configuration:**
   Create or edit the config file with the following content (update the path to match your project location):
   ```json
   {
     "mcpServers": {
       "sqlgenie": {
         "command": "python",
         "args": ["C:\\Users\\krish\\OneDrive\\Desktop\\sharp\\mcp_server.py"],
         "env": {
           "PYTHONPATH": "C:\\Users\\krish\\OneDrive\\Desktop\\sharp"
         }
       }
     }
   }
   ```

   **Important:** Update the paths in the configuration to match your actual project location.

### Step 3: Start the MCP Server
Open VS Code terminal in your project directory and run:
```bash
python mcp_server.py
```

The server will start and wait for Claude Desktop to connect.

### Step 4: Restart Claude Desktop
After updating the configuration, restart Claude Desktop to load the MCP server.

## Available MCP Tools

### 1. `list_databases`
Lists all available databases in your SQLGenie installation.

**Usage in Claude:**
> "Show me all available databases"

### 2. `get_database_schema`
Gets detailed schema information for a specific database.

**Usage in Claude:**
> "Show me the schema for the sakila database"

### 3. `execute_sql_query`
Executes SQL queries on your databases (read-only for security).

**Usage in Claude:**
> "Query the sakila database to find all customers from California"

### 4. `analyze_query_results`
Provides natural language explanations of query results.

**Usage in Claude:**
> "Analyze the results from the previous query"

## Example Conversations with Claude

### Database Exploration
```
You: "What databases do I have available?"
Claude: [Uses list_databases tool] "You have 2 databases available: 
1. Sakila Sample Database (16 tables)
2. sample_company (5 tables)"

You: "Show me the structure of the sakila database"
Claude: [Uses get_database_schema] "The Sakila database contains 16 tables including actor, film, customer, rental, etc. Here's the detailed schema..."
```

### Natural Language Queries
```
You: "Find all customers who rented movies in the last month"
Claude: [Uses execute_sql_query] "I'll query the sakila database for recent rentals..."

You: "What are the top 5 most popular movie categories?"
Claude: [Uses execute_sql_query] "Let me analyze the film categories by rental frequency..."
```

### Data Analysis
```
You: "Give me insights about customer rental patterns"
Claude: [Combines multiple tools] "I'll analyze the rental data to identify patterns..."
```

## Security Features

The MCP server includes several safety measures:

1. **Read-Only Access**: Only SELECT, WITH, PRAGMA, and EXPLAIN queries are allowed
2. **Query Length Limits**: Maximum 10,000 characters per query
3. **Result Limits**: Results are limited to 1000 rows to prevent memory issues
4. **Timeout Protection**: Queries timeout after 30 seconds
5. **Input Validation**: Dangerous operations are blocked

## Troubleshooting

### Common Issues

**1. Claude Desktop doesn't show MCP tools**
- Check that `claude_desktop_config.json` is in the correct location
- Verify the Python path in the configuration is correct
- Restart Claude Desktop after configuration changes

**2. "Command not found" errors**
- Ensure Python is in your system PATH
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify the MCP server script path is correct

**3. Database connection errors**
- Ensure your database files exist in the `data/` folder
- Check file permissions for database files
- Verify the `data/uploaded_databases` and `data/schema_cache` directories exist

**4. Permission errors on Windows**
- Ensure the project directory has proper read/write permissions
- Try running VS Code as Administrator if needed

### Debug Mode
To debug MCP server issues, run it directly in VS Code terminal and check for error messages:
```bash
python mcp_server.py
```

### Test MCP Server
You can test if the MCP server starts correctly:
```bash
python -c "
import subprocess
import sys
import time

# Test MCP server startup
process = subprocess.Popen([sys.executable, 'mcp_server.py'], 
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
time.sleep(2)
if process.poll() is None:
    print('✅ MCP server started successfully')
    process.terminate()
else:
    print('❌ MCP server failed to start')
    print('Error:', process.stderr.read().decode())
"
```

### Logs and Error Messages
The MCP server provides detailed error messages including:
- SQL syntax errors with hints
- Database connection issues
- Security violations
- Performance warnings

## Advanced Usage

### Custom Database Uploads
You can upload new databases through the original SQLGenie web interface, and they'll automatically be available in the MCP server.

### Extending the MCP Server
The MCP server code is designed to be extensible. You can add new tools by:
1. Adding new tool definitions in `handle_list_tools()`
2. Implementing the tool logic in `handle_call_tool()`
3. Following the existing patterns for error handling and validation

### Performance Optimization
- Database schemas are cached for faster access
- Query results are limited to prevent memory issues
- Connection pooling could be added for high-frequency usage

## Integration Benefits

### Compared to Web Interface
- **Faster**: Direct database access without web requests
- **More Natural**: Conversational interface with Claude
- **Integrated**: Works within Claude Desktop's context
- **Powerful**: Combine database queries with Claude's analysis

### Compared to Traditional SQL Tools
- **AI-Powered**: Natural language query generation
- **Intelligent**: Automatic error correction and suggestions
- **Context-Aware**: Claude understands your data and relationships
- **Explanatory**: Get insights, not just raw results

## Support and Contribution

### Getting Help
1. Check this documentation first
2. Review error messages in the MCP server output
3. Verify Claude Desktop logs for connection issues
4. Test with the original SQLGenie web interface to isolate issues

### Contributing
Feel free to enhance the MCP server with additional features:
- More database operations
- Advanced analytics tools
- Data visualization capabilities
- Export/import functionality

---

**Enjoy using SQLGenie with Claude Desktop! 🚀**

## Available MCP Tools

### 1. `list_databases`
Lists all available databases in your SQLGenie installation.

**Usage in Claude:**
> "Show me all available databases"

### 2. `get_database_schema`
Gets detailed schema information for a specific database.

**Usage in Claude:**
> "Show me the schema for the sakila database"

### 3. `execute_sql_query`
Executes SQL queries on your databases (read-only for security).

**Usage in Claude:**
> "Query the sakila database to find all customers from California"

### 4. `analyze_query_results`
Provides natural language explanations of query results.

**Usage in Claude:**
> "Analyze the results from the previous query"

## Example Conversations with Claude

### Database Exploration
```
You: "What databases do I have available?"
Claude: [Uses list_databases tool] "You have 2 databases available: 
1. Sakila Sample Database (16 tables)
2. sample_company (5 tables)"

You: "Show me the structure of the sakila database"
Claude: [Uses get_database_schema] "The Sakila database contains 16 tables including actor, film, customer, rental, etc. Here's the detailed schema..."
```

### Natural Language Queries
```
You: "Find all customers who rented movies in the last month"
Claude: [Uses execute_sql_query] "I'll query the sakila database for recent rentals..."

You: "What are the top 5 most popular movie categories?"
Claude: [Uses execute_sql_query] "Let me analyze the film categories by rental frequency..."
```

### Data Analysis
```
You: "Give me insights about customer rental patterns"
Claude: [Combines multiple tools] "I'll analyze the rental data to identify patterns..."
```

## Security Features

The MCP server includes several safety measures:

1. **Read-Only Access**: Only SELECT, WITH, PRAGMA, and EXPLAIN queries are allowed
2. **Query Length Limits**: Maximum 10,000 characters per query
3. **Result Limits**: Results are limited to 1000 rows to prevent memory issues
4. **Timeout Protection**: Queries timeout after 30 seconds
5. **Input Validation**: Dangerous operations are blocked

## Troubleshooting

### Common Issues

**1. Claude Desktop doesn't show MCP tools**
- Check that `claude_desktop_config.json` is in the correct location
- Verify the Python path in the configuration is correct
- Restart Claude Desktop after configuration changes

**2. "Command not found" errors**
- Ensure Python is in your system PATH
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify the MCP server script path is correct

**3. Database connection errors**
- Ensure your database files exist in the expected locations
- Check file permissions for database files
- Verify the `uploaded_databases` and `schema_cache` directories exist

**4. Permission errors on Windows**
- Run PowerShell as Administrator if needed
- Ensure the project directory has proper read/write permissions

### Debug Mode
To debug MCP server issues, you can run it directly and see error messages:
```bash
python mcp_server.py
```

### Logs and Error Messages
The MCP server provides detailed error messages including:
- SQL syntax errors with hints
- Database connection issues
- Security violations
- Performance warnings

## Advanced Usage

### Custom Database Uploads
You can upload new databases through the original SQLGenie web interface, and they'll automatically be available in the MCP server.

### Extending the MCP Server
The MCP server code is designed to be extensible. You can add new tools by:
1. Adding new tool definitions in `handle_list_tools()`
2. Implementing the tool logic in `handle_call_tool()`
3. Following the existing patterns for error handling and validation

### Performance Optimization
- Database schemas are cached for faster access
- Query results are limited to prevent memory issues
- Connection pooling could be added for high-frequency usage

## Integration Benefits

### Compared to Web Interface
- **Faster**: Direct database access without web requests
- **More Natural**: Conversational interface with Claude
- **Integrated**: Works within Claude Desktop's context
- **Powerful**: Combine database queries with Claude's analysis

### Compared to Traditional SQL Tools
- **AI-Powered**: Natural language query generation
- **Intelligent**: Automatic error correction and suggestions
- **Context-Aware**: Claude understands your data and relationships
- **Explanatory**: Get insights, not just raw results

## Support and Contribution

### Getting Help
1. Check this documentation first
2. Review error messages in the MCP server output
3. Verify Claude Desktop logs for connection issues
4. Test with the original SQLGenie web interface to isolate issues

### Contributing
Feel free to enhance the MCP server with additional features:
- More database operations
- Advanced analytics tools
- Data visualization capabilities
- Export/import functionality

---

**Enjoy using SQLGenie with Claude Desktop! 🚀**