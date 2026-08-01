@echo off
echo ========================================
echo MindBridge Setup Script
echo ========================================
echo.

echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment created
echo.

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat
echo ✅ Virtual environment activated
echo.

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Step 4: Installing dependencies...
echo This may take 5-10 minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ Dependencies installed
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Keep this terminal open
echo 2. Run: python client.py
echo 3. Open browser: http://localhost:5001
echo.
echo Optional: Install Ollama for better responses
echo Download from: https://ollama.com/download
echo Then run: ollama pull llama3
echo.
pause
