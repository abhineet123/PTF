@echo off
python3 "%~dp0\..\rename_subfolders_by_contents.py"  file_ext=%1 folder_name=%2 prefix=%3 exceptions=%4 out_file=%5