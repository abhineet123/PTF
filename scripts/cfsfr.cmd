@echo off
python2 "%~dp0\..\countFileInSubfolders.py" file_ext=%1 prefix=%2 folder_name=%3 recursive=1 shuffle_files=%5 out_file=%6 del_empty=%7