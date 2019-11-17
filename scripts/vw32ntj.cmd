@echo off
python3 "%~dp0\..\visualizeWithMotion.py" on_top=0 top_border=0 keep_borders=1 n_images=2 random_mode=1 auto_progress=1 transition_interval=30 monitor_id=0  duplicate_window=0 second_from_top=1 win_offset_y=0 width=1920 height=1050 src_dirs=20,20/2*5,20/1*3,20/1/1_0*5,20/1/1_1*25,20/1/1_8*8,20/1/1_9*20,20/1/1_3*4,20/1/1_5*15,20/1/1_2*15,20/1/1_4*20,!20/9,!20/10 src_path=%1 n_images=%2 frg_win_titles="The Journal 8","~" only__maximized=0 reversed_pos=0

