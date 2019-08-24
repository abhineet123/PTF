@echo off
python3 "%~dp0\..\visualizeWithMotion.py" on_top=0 top_border=0 keep_borders=1 reversed_pos=2 n_images=2 random_mode=1 auto_progress=1 transition_interval=30 monitor_id=0 dup_monitor_ids=2 duplicate_window=1 second_from_top=2 win_offset_y=0 width=1920 height=1080 src_dirs=20,20/1/1_1*30,20/1/1_3*10,20/1/1_4*10,!20/9,!20/10 src_path=%1 n_images=%2 

