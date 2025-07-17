@echo on
python3 "%~dp0\..\stackVideos.py" fps=5 ext=mp4 codec=avc1 src_paths=%1 grid_size=%2 out_size=%3 annotations=%4  sep_size=%5 n_frames=%6 start_id=%7 
