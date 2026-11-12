@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

:ask_manual
set "ANSWER="
set /p "ANSWER=Fly B1 manually with the keyboard (MANUAL)? [y/n]: "
if /i "!ANSWER!"=="y" (set "MANUAL=1" & set "MIN_FRIENDLY=2") else if /i "!ANSWER!"=="n" (set "MANUAL=0" & set "MIN_FRIENDLY=1") else (echo Please enter y or n. & goto ask_manual)

:ask_friendly
set "N_FRIENDLY="
set /p "N_FRIENDLY=Number of friendly aircraft (N_FRIENDLY, at least !MIN_FRIENDLY!): "
call :check_number "!N_FRIENDLY!" !MIN_FRIENDLY! || goto ask_friendly

:ask_opp
set "N_OPP="
set /p "N_OPP=Number of opponent aircraft (N_OPPONENTS): "
call :check_number "!N_OPP!" 1 || goto ask_opp

if not exist "venv\Scripts\activate.bat" (
    echo venv not found in %CD%\venv
    pause
    exit /b 1
)
call "venv\Scripts\activate.bat"

echo Running test.py with MANUAL=%MANUAL%, N_FRIENDLY=%N_FRIENDLY% and N_OPP=%N_OPP% ...
python test.py %*
set "EXIT_CODE=%ERRORLEVEL%"

pause
exit /b %EXIT_CODE%

:check_number
rem Succeeds if %1 is an integer >= %2.
set "VALUE=%~1"
if "%VALUE%"=="" (echo Please enter a number. & exit /b 1)
for /f "delims=0123456789" %%c in ("%VALUE%") do (echo "%VALUE%" is not a number. & exit /b 1)
if %VALUE% LSS %2 (echo The number must be at least %2. & exit /b 1)
exit /b 0
