@echo off
python "%~dp0\..\imgSeqToVideo.py" src_path=%1 n_frames=%2 width=%3 height=%4 fps=%5 save_path=%6 reverse=1 codec=FFV1 ext=avi