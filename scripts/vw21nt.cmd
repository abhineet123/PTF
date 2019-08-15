@echo off
python3 "%~dp0\..\visualizeWithMotion.py" on_top=0 top_border=0 keep_borders=1 reversed_pos=2 n_images=2 random_mode=1 auto_progress=1 transition_interval=20 monitor_id=4 win_offset_y=0 width=1920 height=1080 src_dirs=20/1/1_1,20/1/1_2,20/1/1_3,20/1/1_4,20/1/1_5,20/1,17/2 src_path=%1 n_images=%2

