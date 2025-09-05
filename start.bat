@echo off
echo Starting the NL to SQL application...
echo.

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting FastAPI backend server...
start "Backend Server" cmd /k "uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo.
echo Starting Streamlit frontend...
start "Frontend App" cmd /k "streamlit run app.py --server.port 8501"

echo.
echo Both servers are starting up...
echo Backend: http://127.0.0.1:8000
echo Frontend: http://localhost:8501
echo.
echo Press any key to exit this window...
pause > nul
