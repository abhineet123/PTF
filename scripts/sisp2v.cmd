@echo off
python3 "%~dp0\..\splitImgSeq.py" video_mode=1 thresh=-2 show_img=0  src_path=%1  show_img=%2 metric=%3 sub_seq_start_id=%4