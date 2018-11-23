@echo off
python "%~dp0\..\stackVideos.py" src_paths=%1 grid_size=%2 annotations=%3 resize_factor=%4 save_path=%5 ann_fmt=%6
