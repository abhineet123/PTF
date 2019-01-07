@echo off
python3 "%~dp0\..\encodeVideo.py" src_path=%1 n_frames=%2 res=%3 start_id=%4 fps=%5 save_path=%6 reverse=1 codec=FFV1 ext=avi out_postfix=rev