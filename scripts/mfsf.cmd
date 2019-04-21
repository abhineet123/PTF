@echo off
python "%~dp0\..\moveFileInSubfolders.py"  file_ext=%1 folder_name=%2 include_folders=%3 prefix=%4 exceptions=%5 out_file=%6