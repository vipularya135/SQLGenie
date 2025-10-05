from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import sqlite3
import google.generativeai as genai
import os
import shutil
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, List

# Directory structure for database storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADED_DBS_DIR = os.path.join(BASE_DIR, 'uploaded_databases')
SCHEMA_CACHE_DIR = os.path.join(BASE_DIR, 'schema_cache')
DEFAULT_DB_PATH = os.path.join(BASE_DIR, 'sakila.db')

# Create directories if they don't exist
os.makedirs(UPLOADED_DBS_DIR, exist_ok=True)
os.makedirs(SCHEMA_CACHE_DIR, exist_ok=True)

# Multiple Gemini API keys for rotation
API_KEYS = [
    'AIzaSyBnKZbUWNjFMjRpbkP2wLz0BuD9qwybg1M',
    'AIzaSyDu20d3hQYVEkqNPdgLaWtLtyH0XiL2Pl0',
    'AIzaSyA49i1bc4iTqRYG3YcOZv617As0FiAqPUA'
]

# Add environment variable support
if os.environ.get('GEMINI_API_KEY'):
    API_KEYS.insert(0, os.environ.get('GEMINI_API_KEY'))

# Initialize with first API key
current_api_index = 0
genai.configure(api_key=API_KEYS[current_api_index])

def rotate_api_key():
    """Rotate to the next available API key"""
    global current_api_index
    current_api_index = (current_api_index + 1) % len(API_KEYS)
    genai.configure(api_key=API_KEYS[current_api_index])
    return current_api_index

def is_rate_limit_error(error_msg: str) -> bool:
    """Check if the error is a rate limit error"""
    error_lower = error_msg.lower()
    return any(phrase in error_lower for phrase in [
        'quota', 'rate limit', 'exceeded', '429', 'too many requests'
    ])

def call_gemini_with_retry(prompt: str, max_retries: int = len(API_KEYS)) -> str:
    """Call Gemini API with automatic key rotation on rate limits"""
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            resp = model.generate_content(prompt)
            return resp.text or ''
        except Exception as e:
            error_msg = str(e)
            if is_rate_limit_error(error_msg) and attempt < max_retries - 1:
                # Rotate to next API key
                new_index = rotate_api_key()
                print(f"Rate limit hit, rotating to API key {new_index + 1}")
                continue
            else:
                # If it's not a rate limit error or we've exhausted all keys, raise the error
                raise e
    
    # This should never be reached, but just in case
    raise Exception("All API keys exhausted")

app = FastAPI()

class QueryIn(BaseModel):
    query: str
    database_id: Optional[str] = None  # If None, use default Sakila DB

class DatabaseInfo(BaseModel):
    id: str
    name: str
    tables: List[str]
    upload_date: str
    file_size: int

class DatabaseUploadResponse(BaseModel):
    database_id: str
    name: str
    tables: List[str]
    message: str

SYSTEM_PROMPT = (
    "You write SQLite queries only. Return just the SQL. No commentary. "
    "For counting total rows across multiple tables, use a simpler approach like counting each table separately or using a single table count."
)

# Complete schema hints for all Sakila tables
SCHEMA_HINT = (
    "-- Tables (SQLite Sakila - Complete Schema):\n"
    "-- actor(actor_id, first_name, last_name, last_update)\n"
    "-- address(address_id, address, address2, district, city_id, postal_code, phone, last_update)\n"
    "-- category(category_id, name, last_update)\n"
    "-- city(city_id, city, country_id, last_update)\n"
    "-- country(country_id, country, last_update)\n"
    "-- customer(customer_id, store_id, first_name, last_name, email, address_id, active, create_date, last_update)\n"
    "-- film(film_id, title, description, release_year, language_id, original_language_id, rental_duration, rental_rate, length, replacement_cost, rating, special_features, last_update)\n"
    "-- film_actor(actor_id, film_id, last_update)\n"
    "-- film_category(film_id, category_id, last_update)\n"
    "-- film_text(film_id, title, description)\n"
    "-- inventory(inventory_id, film_id, store_id, last_update)\n"
    "-- language(language_id, name, last_update)\n"
    "-- payment(payment_id, customer_id, staff_id, rental_id, amount, payment_date, last_update)\n"
    "-- rental(rental_id, rental_date, inventory_id, customer_id, return_date, staff_id, last_update)\n"
    "-- staff(staff_id, first_name, last_name, address_id, picture, email, store_id, active, username, password, last_update)\n"
    "-- store(store_id, manager_staff_id, address_id, last_update)\n"
    "-- Rules:\n"
    "-- - Use only existing columns.\n"
    "-- - Join rental -> inventory -> film when asking about films by rentals.\n"
    "-- - Prefer explicit INNER JOINs, GROUP BY actual selected non-aggregates.\n"
    "-- - Use single quotes for string literals.\n"
    "-- - For table counts, use sqlite_master to get all tables.\n"
)

# A concise few-shot to steer joins correctly
FEW_SHOT = (
    "-- Example NL->SQL:\n"
    "-- NL: List the top 5 most rented films\n"
    "-- SQL: SELECT f.title, COUNT(*) AS rental_count\n"
    "--      FROM rental r\n"
    "--      JOIN inventory i ON r.inventory_id = i.inventory_id\n"
    "--      JOIN film f ON i.film_id = f.film_id\n"
    "--      GROUP BY f.film_id\n"
    "--      ORDER BY rental_count DESC\n"
    "--      LIMIT 5\n"
    "-- NL: How many films are there?\n"
    "-- SQL: SELECT COUNT(*) as total_films FROM film\n"
    "-- NL: Show all tables with row counts\n"
    "-- SQL: SELECT 'actor' AS table_name, COUNT(*) AS row_count FROM actor UNION ALL SELECT 'address', COUNT(*) FROM address UNION ALL SELECT 'category', COUNT(*) FROM category UNION ALL SELECT 'city', COUNT(*) FROM city UNION ALL SELECT 'country', COUNT(*) FROM country UNION ALL SELECT 'customer', COUNT(*) FROM customer UNION ALL SELECT 'film', COUNT(*) FROM film UNION ALL SELECT 'film_actor', COUNT(*) FROM film_actor UNION ALL SELECT 'film_category', COUNT(*) FROM film_category UNION ALL SELECT 'film_text', COUNT(*) FROM film_text UNION ALL SELECT 'inventory', COUNT(*) FROM inventory UNION ALL SELECT 'language', COUNT(*) FROM language UNION ALL SELECT 'payment', COUNT(*) FROM payment UNION ALL SELECT 'rental', COUNT(*) FROM rental UNION ALL SELECT 'staff', COUNT(*) FROM staff UNION ALL SELECT 'store', COUNT(*) FROM store\n"
    "-- NL: Total number of tables in database\n"
    "-- SQL: SELECT COUNT(*) as total_tables FROM sqlite_master WHERE type='table'\n"
)

def generate_database_id(filename: str, content: bytes) -> str:
    """Generate a unique ID for a database based on filename and content hash"""
    content_hash = hashlib.md5(content).hexdigest()[:8]
    clean_name = Path(filename).stem.replace(' ', '_').replace('-', '_')
    return f"{clean_name}_{content_hash}"

def get_database_schema(db_path: str) -> Dict:
    """Extract schema information from a SQLite database"""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get schema for each table
        schema_info = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            schema_info[table] = {
                'columns': [{'name': col[1], 'type': col[2], 'notnull': col[3], 'pk': col[5]} for col in columns],
                'row_count': 0
            }
            
            # Get row count
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                schema_info[table]['row_count'] = cursor.fetchone()[0]
            except:
                schema_info[table]['row_count'] = 0
        
        return {
            'tables': tables,
            'schema': schema_info
        }
    finally:
        conn.close()

def save_database_info(db_id: str, db_name: str, schema_info: Dict):
    """Save database schema information to cache"""
    cache_file = os.path.join(SCHEMA_CACHE_DIR, f"{db_id}.json")
    info = {
        'id': db_id,
        'name': db_name,
        'schema_info': schema_info,
        'upload_date': str(os.path.getctime(os.path.join(UPLOADED_DBS_DIR, f"{db_id}.db"))),
        'file_size': os.path.getsize(os.path.join(UPLOADED_DBS_DIR, f"{db_id}.db"))
    }
    with open(cache_file, 'w') as f:
        json.dump(info, f, indent=2)

def load_database_info(db_id: str) -> Optional[Dict]:
    """Load database information from cache"""
    cache_file = os.path.join(SCHEMA_CACHE_DIR, f"{db_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

def get_database_path(db_id: Optional[str] = None) -> str:
    """Get the path to a database file"""
    if db_id is None:
        return DEFAULT_DB_PATH
    return os.path.join(UPLOADED_DBS_DIR, f"{db_id}.db")

def list_uploaded_databases() -> List[DatabaseInfo]:
    """List all uploaded databases"""
    databases = []
    for filename in os.listdir(SCHEMA_CACHE_DIR):
        if filename.endswith('.json'):
            db_id = filename[:-5]  # Remove .json extension
            info = load_database_info(db_id)
            if info:
                databases.append(DatabaseInfo(
                    id=info['id'],
                    name=info['name'],
                    tables=info['schema_info']['tables'],
                    upload_date=info['upload_date'],
                    file_size=info['file_size']
                ))
    return databases

def validate_sqlite_file(file_content: bytes) -> bool:
    """Validate if the uploaded file is a valid SQLite database"""
    # SQLite files start with "SQLite format 3\000"
    return file_content.startswith(b'SQLite format 3\000')

def generate_schema_hint(db_id: Optional[str] = None) -> str:
    """Generate schema hint for the AI model"""
    if db_id is None:
        # Return the original Sakila schema
        return SCHEMA_HINT
    
    info = load_database_info(db_id)
    if not info:
        return SCHEMA_HINT
    
    schema_info = info['schema_info']['schema']
    hint_lines = ["-- Tables (SQLite Database - Complete Schema):\n"]
    
    for table_name, table_info in schema_info.items():
        columns = [col['name'] for col in table_info['columns']]
        hint_lines.append(f"-- {table_name}({', '.join(columns)})\n")
    
    hint_lines.extend([
        "-- Rules:\n",
        "-- - Use only existing columns.\n",
        "-- - Prefer explicit INNER JOINs, GROUP BY actual selected non-aggregates.\n",
        "-- - Use single quotes for string literals.\n",
        "-- - For table counts, use sqlite_master to get all tables.\n"
    ])
    
    return ''.join(hint_lines)

BLOCKED = [
    'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'ATTACH', 'DETACH', 'PRAGMA'
]

def is_safe_sql(sql: str) -> bool:
    upper = sql.upper()
    return not any(word in upper for word in BLOCKED)


def validate_and_correct_sql(sql: str) -> tuple[str, bool]:
    """Validate SQL and attempt to correct common issues"""
    # Check for common SQL errors
    upper_sql = sql.upper()
    
    # Fix common aggregate function misuse
    if "SUM(COUNT(*))" in upper_sql:
        # Replace with a simpler count approach
        corrected = sql.replace("SUM(COUNT(*))", "COUNT(*)")
        return corrected, True
    
    # Fix mixed query types (COUNT with UNION ALL)
    if "SELECT COUNT(*) AS TOTAL_TABLES" in upper_sql and "UNION ALL" in upper_sql:
        # Split into separate queries - just return the table count part
        return "SELECT COUNT(*) as total_tables FROM sqlite_master WHERE type='table'", True
    
    # Check for other common issues
    if "UNION ALL" in upper_sql and "COUNT(*)" in upper_sql:
        # Simplify complex union count queries
        if "total" in sql.lower() and "row" in sql.lower():
            return "SELECT COUNT(*) FROM film", True
    
    return sql, False


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith('```') and t.endswith('```'):
        lines = [ln for ln in t.splitlines() if not ln.startswith('```')]
        t = "\n".join(lines).strip()
    return t


def to_sql(nl_query: str, database_id: Optional[str] = None) -> str:
    schema_hint = generate_schema_hint(database_id)
    prompt = f"{SYSTEM_PROMPT}\n{schema_hint}\n{FEW_SHOT}\nNL: {nl_query}\nSQL:"
    text = call_gemini_with_retry(prompt)
    text = strip_code_fences(text)
    return text.rstrip(' ;')


def generate_explanation(query: str, sql: str, rows: list) -> str:
    """Generate a natural language explanation of the query results"""
    
    # Prepare the data summary
    if not rows:
        data_summary = "No results were returned."
    else:
        # Show first few rows as examples
        sample_rows = rows[:3] if len(rows) > 3 else rows
        data_summary = f"Found {len(rows)} result(s). Sample data: {sample_rows}"
    
    prompt = f"""You are a helpful database assistant. Explain the results of a database query in natural language.

Original Question: {query}
SQL Query: {sql}
Results: {data_summary}

Please provide a clear, concise explanation of what the results mean in the context of the original question. Focus on insights and patterns in the data. Keep it under 100 words."""

    try:
        return call_gemini_with_retry(prompt)
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


def ai_correct_sql_error(query: str, sql: str, error_msg: str, database_id: Optional[str] = None, attempt: int = 1) -> str:
    """Use Gemini AI to correct SQL errors"""
    
    # Get the appropriate schema information
    if database_id is None:
        schema_text = "Database Schema (SQLite Sakila):\n- actor(actor_id, first_name, last_name, last_update)\n- address(address_id, address, address2, district, city_id, postal_code, phone, last_update)\n- category(category_id, name, last_update)\n- city(city_id, city, country_id, last_update)\n- country(country_id, country, last_update)\n- customer(customer_id, store_id, first_name, last_name, email, address_id, active, create_date, last_update)\n- film(film_id, title, description, release_year, language_id, original_language_id, rental_duration, rental_rate, length, replacement_cost, rating, special_features, last_update)\n- film_actor(actor_id, film_id, last_update)\n- film_category(film_id, category_id, last_update)\n- film_text(film_id, title, description)\n- inventory(inventory_id, film_id, store_id, last_update)\n- language(language_id, name, last_update)\n- payment(payment_id, customer_id, staff_id, rental_id, amount, payment_date, last_update)\n- rental(rental_id, rental_date, inventory_id, customer_id, return_date, staff_id, last_update)\n- staff(staff_id, first_name, last_name, address_id, picture, email, store_id, active, username, password, last_update)\n- store(store_id, manager_staff_id, address_id, last_update)"
    else:
        info = load_database_info(database_id)
        if info:
            schema_info = info['schema_info']['schema']
            schema_lines = ["Database Schema (SQLite):"]
            for table_name, table_info in schema_info.items():
                columns = [col['name'] for col in table_info['columns']]
                schema_lines.append(f"- {table_name}({', '.join(columns)})")
            schema_text = "\n".join(schema_lines)
        else:
            schema_text = "Database Schema: Unable to load schema information"
    
    prompt = f"""You are a SQL expert. Fix the SQL query that has an error.

Original Question: {query}
Failed SQL Query: {sql}
Error Message: {error_msg}
Attempt Number: {attempt}

{schema_text}

Rules:
- Return ONLY the corrected SQL query, no explanations
- Use proper SQLite syntax
- For complex queries, break them into simpler parts
- Use single quotes for string literals
- Ensure all table and column names exist in the schema
- For counting tables, use: SELECT COUNT(*) FROM sqlite_master WHERE type='table'
- For table row counts, use UNION ALL with individual table counts

Corrected SQL:"""

    try:
        corrected_sql = call_gemini_with_retry(prompt).strip()
        
        # Clean up the response
        if corrected_sql.startswith('```'):
            corrected_sql = corrected_sql.split('\n')[1:-1]
            corrected_sql = '\n'.join(corrected_sql)
        
        return corrected_sql.strip()
    except Exception as e:
        return None


def ai_rephrase_question(query: str, correction_history: list) -> str:
    """Use Gemini AI to rephrase a question that failed multiple correction attempts"""
    
    # Summarize the correction attempts
    attempts_summary = ""
    for i, correction in enumerate(correction_history):
        attempts_summary += f"Attempt {i+1}: {correction.get('error', 'Unknown error')}\n"
    
    prompt = f"""You are a helpful database assistant. The user asked a question that couldn't be answered due to technical issues. Please rephrase their question in a simpler, clearer way that would be easier to answer.

Original Question: {query}

Issues encountered:
{attempts_summary}

Database Schema (SQLite Sakila):
- actor, address, category, city, country, customer, film, film_actor, film_category, film_text, inventory, language, payment, rental, staff, store

Please provide 2-3 alternative ways to ask the same question, but simpler and more specific. Focus on:
1. Breaking complex questions into simpler parts
2. Using clearer, more specific language
3. Asking for one thing at a time instead of multiple things

Return the rephrased questions in this format:
1. [First rephrased question]
2. [Second rephrased question]  
3. [Third rephrased question]

Rephrased Questions:"""

    try:
        rephrased = call_gemini_with_retry(prompt).strip()
        
        # Clean up the response
        if rephrased.startswith('```'):
            rephrased = rephrased.split('\n')[1:-1]
            rephrased = '\n'.join(rephrased)
        
        return rephrased.strip()
    except Exception as e:
        return None


def run_sql(sql: str, database_id: Optional[str] = None):
    db_path = get_database_path(database_id)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        cols = list(rows[0].keys()) if rows else [d[0] for d in cur.description] if cur.description else []
        return {"columns": cols, "rows": rows, "sql": sql}
    finally:
        conn.close()


@app.post('/query')
async def query(nl: QueryIn):
    """AI-powered query processing with automatic error correction and retry logic"""
    max_attempts = 3
    current_sql = to_sql(nl.query, nl.database_id)
    original_sql = current_sql
    correction_history = []
    
    # Check safety first
    if not is_safe_sql(current_sql):
        return {"error": "This query contains unsafe operations and cannot be executed.", "sql": current_sql}
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Try to execute the current SQL
            result = run_sql(current_sql, nl.database_id)
            
            # If successful, generate explanation and return
            explanation = generate_explanation(nl.query, current_sql, result.get("rows", []))
            result["explanation"] = explanation
            result["sql"] = current_sql
            
            # Add correction info if any corrections were made
            if correction_history:
                result["ai_corrected"] = True
                result["correction_attempts"] = len(correction_history)
                result["original_sql"] = original_sql
                result["correction_history"] = correction_history
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            # If this is the last attempt, try to rephrase the question
            if attempt == max_attempts:
                # Try to get AI-suggested rephrased questions
                rephrased_questions = ai_rephrase_question(nl.query, correction_history)
                
                return {
                    "error": "Unable to execute the query after multiple correction attempts.",
                    "sql": current_sql,
                    "ai_corrected": True,
                    "correction_attempts": len(correction_history),
                    "original_sql": original_sql,
                    "correction_history": correction_history,
                    "suggested_questions": rephrased_questions,
                    "show_suggestions": True
                }
            
            # Use AI to correct the SQL error
            corrected_sql = ai_correct_sql_error(nl.query, current_sql, error_msg, nl.database_id, attempt)
            
            if corrected_sql and corrected_sql != current_sql:
                # Record the correction
                correction_history.append({
                    "attempt": attempt,
                    "error": error_msg,
                    "original_sql": current_sql,
                    "corrected_sql": corrected_sql
                })
                current_sql = corrected_sql
            else:
                # AI couldn't correct, try a fallback approach
                if "table" in nl.query.lower() and ("count" in nl.query.lower() or "row" in nl.query.lower()):
                    # Special fallback for table counting queries
                    if "total" in nl.query.lower() and "table" in nl.query.lower():
                        current_sql = "SELECT COUNT(*) as total_tables FROM sqlite_master WHERE type='table'"
                    else:
                        # Get available tables for the specific database
                        db_path = get_database_path(nl.database_id)
                        try:
                            schema_info = get_database_schema(db_path)
                            tables = schema_info['tables']
                            union_parts = [f"SELECT '{table}' AS table_name, COUNT(*) AS row_count FROM {table}" for table in tables]
                            current_sql = " UNION ALL ".join(union_parts)
                        except:
                            current_sql = "SELECT COUNT(*) as result FROM sqlite_master WHERE type='table'"
                    
                    correction_history.append({
                        "attempt": attempt,
                        "error": error_msg,
                        "original_sql": current_sql,
                        "corrected_sql": current_sql,
                        "fallback_used": True
                    })
                else:
                    # Generic fallback
                    current_sql = "SELECT COUNT(*) as result FROM sqlite_master WHERE type='table' LIMIT 1"
                    correction_history.append({
                        "attempt": attempt,
                        "error": error_msg,
                        "original_sql": current_sql,
                        "corrected_sql": current_sql,
                        "fallback_used": True
                    })
    
    # This should never be reached, but just in case
    return {"error": "Unexpected error in query processing", "sql": current_sql}

@app.post('/upload-database', response_model=DatabaseUploadResponse)
async def upload_database(file: UploadFile = File(...)):
    """Upload a SQLite database file"""
    
    # Validate file type
    if not file.filename.lower().endswith(('.db', '.sqlite', '.sqlite3')):
        raise HTTPException(status_code=400, detail="Only SQLite database files (.db, .sqlite, .sqlite3) are allowed")
    
    # Read file content
    content = await file.read()
    
    # Validate SQLite format
    if not validate_sqlite_file(content):
        raise HTTPException(status_code=400, detail="Invalid SQLite database file")
    
    # Generate database ID
    db_id = generate_database_id(file.filename, content)
    
    # Check if database already exists
    if load_database_info(db_id):
        info = load_database_info(db_id)
        return DatabaseUploadResponse(
            database_id=db_id,
            name=info['name'],
            tables=info['schema_info']['tables'],
            message="Database already exists and is ready to use"
        )
    
    # Save the database file
    db_path = os.path.join(UPLOADED_DBS_DIR, f"{db_id}.db")
    with open(db_path, 'wb') as f:
        f.write(content)
    
    try:
        # Extract schema information
        schema_info = get_database_schema(db_path)
        
        # Save database info to cache
        save_database_info(db_id, file.filename, schema_info)
        
        return DatabaseUploadResponse(
            database_id=db_id,
            name=file.filename,
            tables=schema_info['tables'],
            message="Database uploaded successfully"
        )
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(db_path):
            os.remove(db_path)
        raise HTTPException(status_code=500, detail=f"Error processing database: {str(e)}")

@app.get('/databases', response_model=List[DatabaseInfo])
async def list_databases():
    """List all uploaded databases"""
    databases = list_uploaded_databases()
    
    # Add default Sakila database if it exists
    if os.path.exists(DEFAULT_DB_PATH):
        try:
            schema_info = get_database_schema(DEFAULT_DB_PATH)
            sakila_info = DatabaseInfo(
                id="sakila",
                name="Sakila (Default)",
                tables=schema_info['tables'],
                upload_date="Built-in",
                file_size=os.path.getsize(DEFAULT_DB_PATH)
            )
            databases.insert(0, sakila_info)
        except:
            pass
    
    return databases

@app.delete('/databases/{database_id}')
async def delete_database(database_id: str):
    """Delete an uploaded database"""
    if database_id == "sakila":
        raise HTTPException(status_code=400, detail="Cannot delete the default Sakila database")
    
    db_path = get_database_path(database_id)
    cache_file = os.path.join(SCHEMA_CACHE_DIR, f"{database_id}.json")
    
    if not os.path.exists(db_path) and not os.path.exists(cache_file):
        raise HTTPException(status_code=404, detail="Database not found")
    
    # Remove files
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(cache_file):
        os.remove(cache_file)
    
    return {"message": "Database deleted successfully"}

@app.get('/databases/{database_id}/schema')
async def get_database_schema_endpoint(database_id: str):
    """Get schema information for a specific database"""
    if database_id == "sakila":
        db_path = DEFAULT_DB_PATH
    else:
        info = load_database_info(database_id)
        if not info:
            raise HTTPException(status_code=404, detail="Database not found")
        db_path = get_database_path(database_id)
    
    try:
        schema_info = get_database_schema(db_path)
        return schema_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading database schema: {str(e)}")
