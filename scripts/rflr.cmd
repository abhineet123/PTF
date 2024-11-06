@echo off
python3 "%~dp0\..\renameFilesFromList.py" dst_first=1 src_names_fname=%1 invert_list=%2 src_root_dir=%3 dst_root_dir=%4