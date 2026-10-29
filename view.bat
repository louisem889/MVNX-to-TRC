@echo off
rem Drag an .mvnx file onto this .bat (optionally together with a .trc) to open the 3D check viewer.
rem Without a .trc it uses converted\<name>.trc.
cd /d "%~dp0"
set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY python --version >nul 2>nul && set "PY=python"
if not defined PY ( echo Python was not found. & pause & exit /b 1 )
%PY% -c "import numpy" >nul 2>nul || %PY% -m pip install --user numpy
if "%~1"=="" (
    echo Drag an .mvnx file ^(and optionally its .trc^) onto view.bat
    pause
    exit /b 0
)
%PY% mvnx_trc_viewer.py %*
if errorlevel 1 pause
