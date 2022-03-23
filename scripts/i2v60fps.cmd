@echo off
python3 "%~dp0\..\imgSeqToVideo.py" fps=60 show_img=0 src_path=%1 n_frames=%2 width=%3 height=%4 codec=%6 save_path=%7 disable_suffix=1