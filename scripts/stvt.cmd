@echo on
python3 "%~dp0\..\stackVideos.py" temporal=1 src_paths=%1 repeat=%2 grid_size=%3 resize_factor=%4 save_path=%5 n_frames=%6 start_id=%7
