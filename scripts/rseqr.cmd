@echo off
python2 "%~dp0\..\renameFilesIntoSeq.py" shuffle_files=1 seq_prefix=%1 seq_root_dir=%2 seq_start_id=%3 filename_fmt=%4 target_ext=%5 write_log=%6

