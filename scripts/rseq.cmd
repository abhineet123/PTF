@echo off
python "%~dp0\..\renameFilesIntoSeq.py" seq_prefix=%1 seq_root_dir=%2 seq_start_id=%3 shuffle_files=%4 filename_fmt=%5 target_ext=%6 write_log=%7

