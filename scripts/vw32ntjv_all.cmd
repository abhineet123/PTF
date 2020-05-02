@echo off
python3 "%~dp0\..\visualizeWithMotion.py" on_top=0 top_border=0 keep_borders=1 n_images=1 random_mode=1 auto_progress=1 fps=24 monitor_id=0  duplicate_window=0 second_from_top=1 win_offset_y=0 width=1920 height=1050 src_dirs=vids/20/2/15,vids/20/2/14*2,vids/20/2/13_wide*4,vids/20/2/13*4,vids/20/2/12*4,vids/0,vids/1***4,vids/2,vids/3 frg_win_titles="The Journal 8","!!!XYplorer 20.10" only_maximized=0 reversed_pos=2 video_mode=3 multi_mode=1 auto_progress_video=1 preserve_order=1 lazy_video_load=0 src_path=%1 n_images=%2 parallel_read=4 reverse_video=1 max_buffer_ram=1e9

REM "x99","grs","orca","x992","grs2","orca2","f","t"
REM !vids/20/2/14/#,vids/20/2/14,vids/20/2/13,
REM vids/17/r2/mkv**1
REM "The Journal 8","!!!XYplorer 20.10"
REM ,vids/4,vids/5,vids/7,vids/8,vids/9,vids/12,vids/13,vids/14,vids/15,vids/16,vids/17***4,vids/18,vids/19,vids/21
REM ,vids/0***2