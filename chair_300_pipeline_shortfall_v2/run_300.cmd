@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=C:\131share\DB\C31\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo ERROR: Python virtual environment not found: %PYTHON%
  exit /b 1
)
"%PYTHON%" run_300_pipeline.py --stage all --resume %*
exit /b %ERRORLEVEL%
