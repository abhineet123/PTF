@echo off
python3 "%~dp0\..\encodeVideo.py" src_path=%1 n_frames=%2 res=1280x720 start_id=%5 save_path=%6 codec=HFYU ext=avi