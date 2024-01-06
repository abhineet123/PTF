@echo off
python3 "%~dp0\..\splitImgSeq.py" metric=0 src_path=%1 thresh=%2 sub_seq_start_id=%3
