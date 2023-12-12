@echo off
python3 "%~dp0\..\encodeVideo.py" codec=H264 ext=mkv src_path=%1 fps=%2 n_frames=%3 res=%4 start_id=%5 reverse=%6 save_path=%8