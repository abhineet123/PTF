@echo off
python3 "%~dp0\..\splitImgSeq.py" src_path=%1 metric=%2 thresh=%3 sub_seq_start_id=%4
