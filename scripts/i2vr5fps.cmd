@echo off
python3 "%~dp0\..\imgSeqToVideo.py" src_path=%1 fps=5 width=%3 height=%4 n_frames=%5 save_path=%6 reverse=1 codec=mp4v ext=mp4