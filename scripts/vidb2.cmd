@echo off
python3 "%~dp0\..\imgSeqToVideo.py" save_root_dir=L:/vids/20/2 codec=FFV1 ext=avi show_img=0 src_path=%1 fps=%2 n_frames=%3 width=%4 height=%5 save_path=%6 