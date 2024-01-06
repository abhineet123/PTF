@echo on
python3 "%~dp0\..\stackVideos.py" fps=1 src_paths=%1 grid_size=%2 annotations=%3 out_size=%4 sep_size=%5 n_frames=%6 start_id=%7
