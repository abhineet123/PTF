@echo off
python2 "%~dp0\..\videoToImgSeq.py"  seq_name=%1 reverse=%2 n_frames=%3 start_id=%4 dst_dir=%5 reverse=0 ext=png 