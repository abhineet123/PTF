@echo off
python2 "%~dp0\..\renameFilesIntoSeq.py" recursive=1 seq_prefix=%1 seq_prefix_filter=%2 seq_root_dir=%3 seq_start_id=%4 shuffle_files=%5 filename_fmt=%6 target_ext=%7 write_log=%7

