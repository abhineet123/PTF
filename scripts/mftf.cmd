
@echo off
REM use mftsf for moving each file to separate folder
python3 "%~dp0\..\moveFilesToSubDirs.py" subdir_prefix=%1 files_per_subdir=%2 root_dir=%3 rename_files=%4 subdir_start_id=%5 shuffle_files=%6

