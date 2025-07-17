@echo off
python3 "%~dp0\..\visualizeWithMotion.py" src_path=%1 n_images=%2 lazy_video_load=0 fullscreen=1 exclude_src_dirs="masks,contours" trim_images=0 reverse_video=0 random_mode=1 auto_progress=1 transition_interval=60 monitor_scale=1.5

