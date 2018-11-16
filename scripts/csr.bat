@echo off
set test=%~1
python3 "%~dp0\..\consolidateResults.py" list_file=. dir_pattern=%test% template_id=%2