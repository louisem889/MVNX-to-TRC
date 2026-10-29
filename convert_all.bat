@echo off
rem Converts every .mvnx in this folder (or the files/folders dropped onto this .bat)
rem to .trc. Results go into the "converted" subfolder (re-running replaces them).
cd /d "%~dp0"

rem --- find Python (py launcher first, then python) ---
set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY python --version >nul 2>nul && set "PY=python"
if not defined PY (
    echo.
    echo Python was not found. Install it from the Microsoft Store ^(search "Python"^)
    echo or from https://www.python.org/downloads/ and run this file again.
    echo.
    pause
    exit /b 1
)

rem --- install numpy the first time ---
%PY% -c "import numpy" >nul 2>nul || (
    echo Installing numpy ^(one-time^)...
    %PY% -m pip install --user numpy
)

set "HARG="
set "V="
set /p "V=Also create a 3D check viewer for each file? (y/N): "
if /i "%V%"=="y" set "HARG=%HARG% --viewer"

if "%~1"=="" (
    %PY% mvnx_to_trc.py convert "%~dp0." -o "%~dp0converted" --overwrite %HARG%
) else (
    %PY% mvnx_to_trc.py convert %* -o "%~dp0converted" --overwrite %HARG%
)
pause
