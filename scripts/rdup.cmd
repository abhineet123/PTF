@echo off
python3 "%~dp0\..\findDuplicateFilesByHashPairs.py" root_dir=%1 file_type=%2 delete_file=1 db_file=fdup_db.pkl