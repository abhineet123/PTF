@echo off
python3 "%~dp0\..\findDuplicateFilesByHashPairs.py" file_type=%1 root_dir=%2 delete_file=0 db_file=fdup_db.pkl