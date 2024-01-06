@echo off
python3 "%~dp0\..\encodeVideo.py" src_path=%1 fps=%2 res=%3 start_id=%4 n_frames=%5 save_path=%6 reverse=2 codec=FFV1 ext=avi