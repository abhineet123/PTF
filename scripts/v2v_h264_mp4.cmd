@echo off
python3 "%~dp0\..\encodeVideo.py" codec=avc1 ext=mp4 fps=30 src_path=%1 fps=%2 n_frames=%3 res=%4 start_id=%5 reverse=%6 save_path=%8