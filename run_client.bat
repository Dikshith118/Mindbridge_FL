@echo off
echo ========================================
echo Starting MindBridge Local Client
echo ========================================
echo.

if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting local client server...
echo Your data stays on YOUR machine in: client_data/
echo.
echo ========================================
echo Open your browser and go to:
echo http://localhost:5001
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

python client.py

pause
