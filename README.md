# SQLGenie - AI-Powered Natural Language to SQL Converter with Dynamic Database Support
Transform your natural language questions into perfect SQL queries with intelligent error correction and smart suggestions. Upload your own SQLite databases or use the built-in Sakila sample database.
## Features
- **🗂️ Dynamic Database Support** - Upload your own SQLite database files (.db, .sqlite, .sqlite3)
- **🧠 Automatic Schema Discovery** - Automatically extracts and caches database schema information
- **📚 Database Management** - List, select, and delete uploaded databases
- **🔄 Database Caching** - Stores uploaded databases and schemas for future reuse
- Convert natural language queries to SQL
- **Natural language explanations** of query results using AI
- **🤖 AI-powered error correction** - Automatically fixes SQL errors using Gemini AI
- **Intelligent retry logic** - Up to 3 attempts with AI corrections
- **Correction history tracking** - See exactly how the AI fixed your query
- **💡 Smart question rephrasing** - AI suggests better ways to ask questions when corrections fail
- **🔄 API Key Rotation** - Automatically switches between multiple Gemini API keys on rate limits
- **Technical error visibility** (shows actual error messages for debugging)
- Interactive web interface with Streamlit
- FastAPI backend for processing queries
- Built-in examples and query history
- CSV export functionality
## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Running the Application
### Option 1: Use the startup script (Recommended)
Double-click `start.bat` or run `start.ps1` in PowerShell.
### Option 2: Manual startup
1. Start the backend server:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```
2. In a new terminal, start the frontend:
   ```bash
   streamlit run app.py --server.port 8501
   ```
## Usage
1. Open your browser and go to `http://localhost:8501`
2. **Upload a Database** (optional):
   - Click "Browse files" in the sidebar under "Upload Database"
   - Select a SQLite database file (.db, .sqlite, .sqlite3)
   - Click "Upload Database" to process and cache the database
3. **Select a Database**:
   - Choose from uploaded databases or use the default Sakila database
   - View table information for the selected database
4. Enter a natural language query in the text box
5. Click "🔮 Generate SQL" to execute the query
6. View the generated SQL, natural language explanation, and results
7. Download results as CSV if needed
8. **Manage Databases**: Delete uploaded databases when no longer needed
### What You'll See
- **🗂️ Database Management**: Upload, select, and delete SQLite databases
- **📊 Schema Information**: View tables and structure for selected databases
- **🔄 Database Caching**: Automatic storage and reuse of uploaded databases
- **Generated SQL**: The SQL query created from your natural language input
- **🤖 AI Corrections**: Automatic fixes for SQL errors using AI (with success notifications)
- **Correction History**: Detailed view of how the AI fixed your query step-by-step
- **💡 Question Suggestions**: AI suggests better ways to ask questions when all corrections fail
- **Results Explanation**: An AI-generated explanation of what the results mean
- **Data Table**: The actual query results in a formatted table
- **CSV Download**: Download results as CSV files
- **Query History**: Previous queries with their explanations and database context
- **Smart Error Handling**: AI-powered error correction with retry logic
- **Technical Error Display**: Shows actual error messages for debugging and learning
- **API Key Management**: Automatic rotation between multiple API keys to handle rate limits
## Example Queries
- "List the top 5 most rented films"
- "Find customers from Germany who spent more than $50"
- "Show total revenue by country (top 10)"
- "Which staff processed the most rentals?"
## Database Support
### Default Database
The application includes the Sakila sample database, which contains information about:
- Films and inventory
- Customer and rental data
- Payment information
- Staff and store details
- Geographic data (cities, countries, addresses)

### Custom Databases
- **Upload Support**: Upload any SQLite database file (.db, .sqlite, .sqlite3)
- **Automatic Schema Detection**: Automatically discovers tables, columns, and relationships
- **Intelligent Caching**: Stores uploaded databases and their schemas for future use
- **Multi-Database Support**: Switch between different databases seamlessly
- **Database Management**: View, select, and delete uploaded databases

### Supported File Types
- `.db` - SQLite database files
- `.sqlite` - SQLite database files
- `.sqlite3` - SQLite database files

### File Validation
- Validates SQLite file format before processing
- Checks database integrity and accessibility
- Generates unique identifiers for each uploaded database
- Prevents duplicate uploads of the same database