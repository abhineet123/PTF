@echo off
python3 "%~dp0\..\combine_csv_files.py"  file_list=%1 root_dir=%2 dst_path=%3