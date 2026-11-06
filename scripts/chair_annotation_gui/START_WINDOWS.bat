@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if not errorlevel 1 goto use_py
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if not errorlevel 1 goto use_python
echo Python 3.10 or newer is required.
echo Install Python from https://www.python.org/downloads/
echo Then open START_WINDOWS.bat again.
pause
popd
exit /b 1
:use_py
py -3 -X utf8 "%~dp0launcher.py" %*
goto finished
:use_python
python -X utf8 "%~dp0launcher.py" %*
:finished
if errorlevel 1 pause
popd
endlocal
