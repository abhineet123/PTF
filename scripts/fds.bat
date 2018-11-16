@echo off

REM For /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
REM For /f "tokens=1-2 delims=/:" %%a in ("%TIME%") do (set mytime=%%a%%b)
REM echo %mydate%_%mytime%

python3 "%~dp0\..\printDateTime.py" prefix=%1 file_path=%2