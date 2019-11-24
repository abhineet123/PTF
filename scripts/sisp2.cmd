@echo off
python3 "%~dp0\..\splitImgSeq.py" thresh=-2 src_path=%1 metric=%2 sub_seq_start_id=%3
