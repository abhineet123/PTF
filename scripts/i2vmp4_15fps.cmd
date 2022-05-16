@echo off
python3 "%~dp0\..\imgSeqToVideo.py" show_img=0 src_path=%1 width=%2 height=%3 fps=15 n_frames=%5 codec=%6 save_path=%7 ext=mp4 codec=H264 size=1920x1080