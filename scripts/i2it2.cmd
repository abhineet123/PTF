@echo off
python3 "%~dp0\..\videoToImgSeq.py" tracker_type=1 crop=2 mode=1 show_img=1 seq_name=%1 tracker_type=%2 start_id=%3 dst_dir=%4 reverse=%5