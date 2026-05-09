@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo ReaDirect-TTS virtual environment was not found.
    echo Create it with: py -m venv .venv
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m uvicorn tts_service:app --host 127.0.0.1 --port 8002

if errorlevel 1 (
    echo.
    echo ReaDirect-TTS stopped with an error.
    pause
)
