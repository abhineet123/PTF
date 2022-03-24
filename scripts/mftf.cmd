@echo off
python3 "%~dp0\..\moveFilesToSubDirs.py" subdir_prefix=%1 files_per_subdir=%2 shuffle_files=%3  rename_files=%4 subdir_start_id=%5 root_dir=%6 

