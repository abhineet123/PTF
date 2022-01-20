@echo on
python3 "%~dp0\..\stackVideosMulti.py" fps=1 ext=mkv root_dirs=%1 grid_size=%2 annotations=%3 resize_factor=%4 save_path=%5 n_frames=%6 start_id=%7
