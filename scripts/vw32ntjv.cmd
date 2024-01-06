@echo off
python3 "%~dp0\..\visualizeWithMotion.py" on_top=0 top_border=0 keep_borders=1 n_images=1 random_mode=1 auto_progress=1 fps=30 monitor_id=0  duplicate_window=0 second_from_top=1 win_offset_y=0 width=1920 height=1050 src_dirs=vids/20/2/15,vids/20/2/14*2,vids/20/2/13*4,vids/20/2/12*8 frg_win_titles="The Journal 8" only_maximized=0 reversed_pos=2 video_mode=1 multi_mode=1 auto_progress_video=1 reverse_video=1 preserve_order=1 lazy_video_load=0 parallel_read=0 frg_monitor_ids=0,1,2,4,5 only_maximized=0 reversed_pos=2 min_aspect_ratio=0.3 auto_min_aspect_ratio=0.25 max_aspect_ratio=1.8 auto_aspect_ratio=1 max_magnified_height_ratio=2 src_dirs=%1 n_images=%2

REM "x99","grs","orca","x992","grs2","orca2","f","t"
REM !vids/20/2/14/#,vids/20/2/14,vids/20/2/13,
REM vids/17/r2/mkv**1
REM "The Journal 8","!!!XYplorer 20.10"