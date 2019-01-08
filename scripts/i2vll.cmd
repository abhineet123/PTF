@echo off
python "%~dp0\..\imgSeqToVideo.py" codec=FFV1 ext=avi src_path=%1 fps=%2 n_frames=%3 width=%4 height=%5 save_path=%6