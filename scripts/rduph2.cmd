@echo off
python3 "%~dp0\..\findDuplicateFilesByHashPairs.py" files=. delete_file=2 db_file=fdup_db.pkl file_type=img root_dir=%1 show_img=%2 file_type=%3