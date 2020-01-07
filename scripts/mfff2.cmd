@echo off
python2 "%~dp0\..\moveFileIFromSubfolders.py"  include_folders=2 file_ext=%1 folder_name=%2  prefix=%4 exceptions=%5 out_file=%6