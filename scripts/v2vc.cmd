@echo off
python3 "%~dp0\..\videoToImgSeq.py" crop=1 mode=0 show_img=1 seq_name=%1 n_frames=%3 start_id=%2 dst_dir=%4 reverse=%5