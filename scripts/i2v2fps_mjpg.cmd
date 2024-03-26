@echo off
python3 "%~dp0\..\imgSeqToVideo.py" fps=2 codec=MJPG ext=avi show_img=0 src_path=%1 n_frames=%2 width=%3 height=%4 save_path=%5