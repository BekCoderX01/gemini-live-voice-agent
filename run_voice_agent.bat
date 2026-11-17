@echo off
title Gemini Live Voice Agent
cd /d "%~dp0"
echo ========================================================
echo   Starting Gemini Live Voice Agent...
echo   Open your browser at: http://127.0.0.1:8000
echo ========================================================
echo.
.\.venv\Scripts\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
