@echo off
python3 "%~dp0\..\imgSeqToVideo.py" fps=5 show_img=0 write_filename=1 src_path=%1 n_frames=%2 width=1280 height=720 codec=%6 save_path=%7 ext=mp4