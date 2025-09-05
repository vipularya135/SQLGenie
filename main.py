# ...existing code...
def ai_rephrase_question(query: str, correction_history: list) -> str:
    """Use Gemini or MCP to rephrase a question that failed multiple correction attempts"""
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
        if USE_MCP:
            rephrased = call_mcp(prompt).strip()
        else:
            rephrased = call_gemini_with_retry(prompt).strip()
        # Clean up the response
        if rephrased.startswith('```'):
            rephrased = rephrased.split('\n')[1:-1]
            rephrased = '\n'.join(rephrased)
        return rephrased.strip()
    except Exception as e:
        return None
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import google.generativeai as genai
import os
def get_mcp_llm():
    if MCP_API_KEY:
        return OpenAI(openai_api_key=MCP_API_KEY, temperature=0)
    else:
        return OpenAI(temperature=0)

# MCP/LangChain equivalents
def call_mcp(prompt: str) -> str:
    llm = get_mcp_llm()
    return llm(prompt)
# MCP/LangChain config
USE_MCP = os.environ.get('USE_MCP', 'false').lower() == 'true'
MCP_API_KEY = os.environ.get('OPENAI_API_KEY', None)

# Initialize LangChain MCP (OpenAI)
def get_mcp_llm():
    if MCP_API_KEY:
        return OpenAI(openai_api_key=MCP_API_KEY, temperature=0)
    else:
        return OpenAI(temperature=0)


# MCP/LangChain imports
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# MCP/LangChain config
USE_MCP = os.environ.get('USE_MCP', 'false').lower() == 'true'
MCP_API_KEY = os.environ.get('OPENAI_API_KEY', None)

# Path to the existing SQLite DB
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sakila.db')

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
            model = genai.GenerativeModel('gemini-1.5-flash')
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


def to_sql(nl_query: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n{SCHEMA_HINT}\n{FEW_SHOT}\nNL: {nl_query}\nSQL:"
    if USE_MCP:
        text = call_mcp(prompt)
    else:
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
        if USE_MCP:
            return call_mcp(prompt)
        else:
            return call_gemini_with_retry(prompt)
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


def ai_correct_sql_error(query: str, sql: str, error_msg: str, attempt: int = 1) -> str:
    """Use Gemini AI to correct SQL errors"""

    prompt = f"""You are a SQL expert. Fix the SQL query that has an error.

Original Question: {query}
Failed SQL Query: {sql}
Error Message: {error_msg}
Attempt Number: {attempt}

Database Schema (SQLite Sakila):

Rules:

Corrected SQL:"""
    try:
        if USE_MCP:
            corrected_sql = call_mcp(prompt).strip()
        else:
            corrected_sql = call_gemini_with_retry(prompt).strip()
        if corrected_sql.startswith('```'):
            corrected_sql = corrected_sql.split('\n')[1:-1]
            corrected_sql = '\n'.join(corrected_sql)
        return corrected_sql.strip()
    except Exception as e:
        return None





def run_sql(sql: str):
    conn = sqlite3.connect(DB_PATH)
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

    # Special handling for compound table/row count questions
    q_lower = nl.query.lower()
    if ("table" in q_lower and ("count" in q_lower or "row" in q_lower)) and ("total" in q_lower or "each" in q_lower):
        # Run two queries and combine results
        total_tables_sql = "SELECT COUNT(*) as total_tables FROM sqlite_master WHERE type='table'"
        row_counts_sql = "SELECT 'actor' AS table_name, COUNT(*) AS row_count FROM actor UNION ALL SELECT 'address', COUNT(*) FROM address UNION ALL SELECT 'category', COUNT(*) FROM category UNION ALL SELECT 'city', COUNT(*) FROM city UNION ALL SELECT 'country', COUNT(*) FROM country UNION ALL SELECT 'customer', COUNT(*) FROM customer UNION ALL SELECT 'film', COUNT(*) FROM film UNION ALL SELECT 'film_actor', COUNT(*) FROM film_actor UNION ALL SELECT 'film_category', COUNT(*) FROM film_category UNION ALL SELECT 'film_text', COUNT(*) FROM film_text UNION ALL SELECT 'inventory', COUNT(*) FROM inventory UNION ALL SELECT 'language', COUNT(*) FROM language UNION ALL SELECT 'payment', COUNT(*) FROM payment UNION ALL SELECT 'rental', COUNT(*) FROM rental UNION ALL SELECT 'staff', COUNT(*) FROM staff UNION ALL SELECT 'store', COUNT(*) FROM store"
        total_tables_result = run_sql(total_tables_sql)
        row_counts_result = run_sql(row_counts_sql)
        return {
            "sql": f"{total_tables_sql};\n{row_counts_sql}",
            "total_tables": total_tables_result.get("rows", []),
            "row_counts": row_counts_result.get("rows", []),
            "explanation": "This combines the total number of tables and the row count for each table in the Sakila database.",
        }

    # ...existing code...
    max_attempts = 3
    current_sql = to_sql(nl.query)
    original_sql = current_sql
    correction_history = []
    # Check safety first
    if not is_safe_sql(current_sql):
        return {"error": "This query contains unsafe operations and cannot be executed.", "sql": current_sql}
    for attempt in range(1, max_attempts + 1):
        try:
            # Try to execute the current SQL
            result = run_sql(current_sql)
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
            corrected_sql = ai_correct_sql_error(nl.query, current_sql, error_msg, attempt)
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
                        current_sql = "SELECT 'actor' AS table_name, COUNT(*) AS row_count FROM actor UNION ALL SELECT 'address', COUNT(*) FROM address UNION ALL SELECT 'category', COUNT(*) FROM category UNION ALL SELECT 'city', COUNT(*) FROM city UNION ALL SELECT 'country', COUNT(*) FROM country UNION ALL SELECT 'customer', COUNT(*) FROM customer UNION ALL SELECT 'film', COUNT(*) FROM film UNION ALL SELECT 'film_actor', COUNT(*) FROM film_actor UNION ALL SELECT 'film_category', COUNT(*) FROM film_category UNION ALL SELECT 'film_text', COUNT(*) FROM film_text UNION ALL SELECT 'inventory', COUNT(*) FROM inventory UNION ALL SELECT 'language', COUNT(*) FROM language UNION ALL SELECT 'payment', COUNT(*) FROM payment UNION ALL SELECT 'rental', COUNT(*) FROM rental UNION ALL SELECT 'staff', COUNT(*) FROM staff UNION ALL SELECT 'store', COUNT(*) FROM store"
                    correction_history.append({
                        "attempt": attempt,
                        "error": error_msg,
                        "original_sql": current_sql,
                        "corrected_sql": current_sql,
                        "fallback_used": True
                    })
                else:
                    # Generic fallback
                    current_sql = "SELECT COUNT(*) as result FROM film LIMIT 1"
                    correction_history.append({
                        "attempt": attempt,
                        "error": error_msg,
                        "original_sql": current_sql,
                        "corrected_sql": current_sql,
                        "fallback_used": True
                    })
    # This should never be reached, but just in case
    return {"error": "Unexpected error in query processing", "sql": current_sql}
