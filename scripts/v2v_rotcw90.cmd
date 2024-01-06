@echo off
python3 "%~dp0\..\encodeVideo.py" src_path=%1 fps=%2 n_frames=%3 res=%4 start_id=%5 reverse=%6 codec=mp4v save_path=%8 rotate=1 use_skv=0