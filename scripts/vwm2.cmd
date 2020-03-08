@echo off
python3 "%~dp0\..\visualizeWithMotion.py" src_path=%1 random_mode=0 video_mode=2 preserve_order=1 parallel_read=4

