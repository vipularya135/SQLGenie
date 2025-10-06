# SQLGenie - AI-Powered Natural Language to SQL Converter with MCP Integration

Transform your natural language questions into perfect SQL queries with intelligent error correction and smart suggestions. Now with **Claude Desktop integration** through Model Context Protocol (MCP)!

## 🚀 Features

### Core Features
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

### New: MCP Integration 🆕
- **🤖 Claude Desktop Integration** - Direct database access through Claude Desktop
- **🗣️ Natural Language Interface** - Ask Claude about your databases in plain English
- **🔒 Secure Read-Only Access** - Safe database exploration with built-in protections
- **⚡ Real-Time Schema Discovery** - Claude can explore your database structure instantly
- **🆓 Completely Free** - No additional API costs when using Claude Desktop

## 📁 Project Structure

```
SQLGenie/
├── 📄 Core Application Files
│   ├── app.py              # Streamlit frontend
│   ├── main.py             # FastAPI backend
│   ├── mcp_server.py       # MCP server for Claude Desktop
│   └── requirements.txt    # Python dependencies
│
├── 📊 data/                # Database and data files
│   ├── sakila.db          # Sample database
│   ├── sample_company.sql # SQL dump file
│   ├── uploaded_databases/ # User uploaded databases
│   └── schema_cache/      # Cached database schemas
│
├──  docs/               # Documentation
│   └── MCP_INTEGRATION_GUIDE.md # Complete MCP setup guide
│
└── 🔄 Environment
    └── .venv/             # Virtual environment
```

## ⚡ Quick Start

### Option 1: Use with Claude Desktop (Recommended)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Claude Desktop:**
   
   Find your Claude Desktop config file:
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

   Add this configuration (update the path to match your project location):
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

3. **Start MCP Server:**
   ```bash
   python mcp_server.py
   ```

4. **Restart Claude Desktop** and start chatting with your databases!

### Option 2: Use Web Interface

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Backend (Terminal 1):**
   ```bash
   uvicorn main:app --reload
   ```

3. **Start Frontend (Terminal 2):**
   ```bash
   streamlit run app.py
   ```

4. **Open browser:** `http://localhost:8501`

## 💬 Usage Examples

### With Claude Desktop
```
"What databases do I have available?"
"Show me the schema of the sakila database"
"Find all customers from California in the sakila database"
"What are the most popular movie categories?"
"Analyze rental patterns in my database"
```

### With Web Interface
1. Upload a database or use the built-in Sakila database
2. Enter natural language queries like:
   - "Show me all customers from California"
   - "What are the top 5 most rented movies?"
   - "Find all overdue rentals"

## 🔧 Technical Details

### MCP Integration
- **Server:** `mcp_server.py` provides 4 MCP tools for database operations
- **Security:** Read-only access, query limits, input validation
- **Performance:** Result limits (1000 rows), query timeouts (30s)

### Web Application
- **Frontend:** Streamlit with file upload and result visualization
- **Backend:** FastAPI with automatic error correction
- **AI:** Google Gemini with multi-key rotation for reliability

## 🛠️ Development Commands

### Testing MCP Server
```bash
# Test MCP server functionality
python -c "
import subprocess
import sys
import time

# Start MCP server test
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
"
```

### Environment Setup
```bash
# Create virtual environment (optional)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Troubleshooting

### MCP Issues
- Ensure Claude Desktop config file has correct paths
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify MCP server starts without errors: `python mcp_server.py`
- Restart Claude Desktop after configuration changes

### Web Interface Issues
- Verify ports 8000 and 8501 are available
- Check that database files exist in `data/` folder
- Ensure Gemini API keys are valid in `main.py`

### Database Path Issues
- Database files should be in the `data/` folder
- Check that `data/sakila.db` exists
- Verify `data/uploaded_databases/` directory permissions

## 📖 Documentation

- **Complete MCP Guide:** `docs/MCP_INTEGRATION_GUIDE.md`
- **API Documentation:** Available at `http://localhost:8000/docs` when backend is running
- **Example Queries:** Built into the web interface

## 🤝 Contributing

Feel free to contribute by:
- Adding new MCP tools
- Improving database support  
- Enhancing the AI query generation
- Adding new database formats

---

**Transform your database interactions with AI! 🚀**

## 💬 Usage Examples

### With Claude Desktop
```
"What databases do I have available?"
"Show me the schema of the sakila database"
"Find all customers from California in the sakila database"
"What are the most popular movie categories?"
"Analyze rental patterns in my database"
```

### With Web Interface
1. Upload a database or use the built-in Sakila database
2. Enter natural language queries like:
   - "Show me all customers from California"
   - "What are the top 5 most rented movies?"
   - "Find all overdue rentals"

## 🔧 Technical Details

### MCP Integration
- **Server:** `mcp_server.py` provides 4 MCP tools for database operations
- **Security:** Read-only access, query limits, input validation
- **Performance:** Result limits (1000 rows), query timeouts (30s)

### Web Application
- **Frontend:** Streamlit with file upload and result visualization
- **Backend:** FastAPI with automatic error correction
- **AI:** Google Gemini with multi-key rotation for reliability

## 📖 Documentation

- **Complete MCP Guide:** `docs/MCP_INTEGRATION_GUIDE.md`
- **API Documentation:** Available at `http://localhost:8000/docs` when running
- **Example Queries:** Built into the web interface

## 🛠️ Troubleshooting

### MCP Issues
- Run `.\scripts\test_mcp.py` to verify setup
- Check Claude Desktop config: `%APPDATA%\Claude\claude_desktop_config.json`
- Ensure all dependencies are installed

### Web Interface Issues
- Verify ports 8000 and 8501 are available
- Check that database files exist in `data/` folder
- Ensure Gemini API keys are valid

## 🤝 Contributing

Feel free to contribute by:
- Adding new MCP tools
- Improving database support
- Enhancing the AI query generation
- Adding new database formats

---

**Transform your database interactions with AI! 🚀**
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