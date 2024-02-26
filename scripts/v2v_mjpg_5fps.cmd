@echo off
python3 "%~dp0\..\encodeVideo.py" src_path=%1 fps=5 n_frames=%3 res=%4 start_id=%5 reverse=%6 codec=MJPG ext=avi save_path=%8