@echo off
python "%~dp0\..\renameFilesIntoSeq.py" image seq_prefix=1 seq_start_id=0 shuffle_files=1 seq_root_dir=%1 target_ext=%2 
