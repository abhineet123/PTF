@echo off
python3 "%~dp0\..\encodeVideo.py" res=1920x1080 codec=mp4v ext=mp4 fps=30 src_path=%1 fps=%2 n_frames=%3 start_id=%4 reverse=%5 save_path=%6