@echo off
python3 "%~dp0\..\renameFilesIntoSeq.py" seq_prefix=%1 seq_prefix_filter=%2 seq_start_id=%3 shuffle_files=%4 filename_fmt=%5 target_ext=%6 write_log=%7

