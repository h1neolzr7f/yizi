@echo off
cd /d "%~dp0"
title AI漫画编剧大师
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m yizi.launch
  exit /b 0
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m yizi.launch
) else (
  python -m yizi.launch
)
if errorlevel 1 pause
