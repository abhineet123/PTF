@echo off

FOR /f %%p in ('python3 -c "import os, sys; print(os.path.normpath(sys.prefix))"') do SET PYTHON_PATH=%%p
ECHO %PYTHON_PATH%

python3 "%PYTHON_PATH%\Tools\scripts\2to3.py" -w %1

