@echo off
python3 "%~dp0\..\countFileInSubfolders.py" sort_by_count=1 prefix=%1 file_ext=%2 folder_name=%3 recursive=0 shuffle_files=%5 out_file=%6 del_empty=%7