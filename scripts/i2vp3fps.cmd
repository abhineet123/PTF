@echo off
python3 "%~dp0\..\imgSeqToVideo.py" fps=0.3 ext=mp4 codec=mp4v show_img=0 write_filename=0 src_path=%1 width=%2 height=%3