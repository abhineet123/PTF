@echo off
python3 "%~dp0\..\imgSeqToVideo.py" show_img=0 src_path=%1 size=%2 fps=%3 n_frames=%4 codec=%6 save_path=%7 ext=avi codec=mjpg