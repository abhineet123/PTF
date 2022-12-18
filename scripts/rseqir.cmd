@echo off
python3 "%~dp0\..\renameFilesIntoSeq.py" seq_prefix=image filename_fmt=1 seq_start_id=1 shuffle_files=1 seq_root_dir=%1 target_ext=%2 
