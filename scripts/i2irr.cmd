@echo off

python3 "%~dp0\..\videoToImgSeq.py" mode=1 show_img=1 seq_name=%1 n_frames=%2 start_id=%3 dst_dir=%4 reverse=2