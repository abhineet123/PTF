@echo off
python2 "%~dp0\..\renameFilesIntoSeq.py" recursive=1 seq_prefix=%1 seq_prefix_ext=%2 seq_prefix_filter=%3 seq_root_dir=%4 seq_start_id=%5 shuffle_files=%6 filename_fmt=%7 target_ext=%8 write_log=%9

