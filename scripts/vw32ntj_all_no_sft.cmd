@echo off
python3 "%~dp0\..\visualizeWithMotion.py" on_top=0 top_border=0 keep_borders=1 n_images=1 random_mode=1 auto_progress=1 transition_interval=15 monitor_id=0  duplicate_window=0 second_from_top=0 win_offset_y=0 width=1920 height=1050 src_dirs=!!#,!!##,!!#bad,!!#proc,!!#seg_masks,!!#masks,!!#contours,24/1*2,24/2*6,24*8,25*6,26*4,27*2,18*2,19*2,21*2,28*2,29*3,17/2 min_aspect_ratio=0 auto_min_aspect_ratio=0 max_aspect_ratio=0.25 auto_aspect_ratio=0 max_magnified_height_ratio=2 only_maximized=0 reversed_pos=2 src_dirs=%1 hide_border=0 resizable=1 write_filenames=1

REM "x99","grs","orca","x992","grs2","orca2","f","t"
REM frg_win_titles="The Journal 8","!!!XYplorer 20.10"
REM ,"PyCharm"
REM vids/20/2/13**20,vids/20/2_patches**20
REM src_dirs=20/0,20/1*5,20/2*5,20/3,20/8,20/12,17/2*5,17/3*3,17/4,17/5,17/7_3,#patches_src,0***20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,21,24,22***1,23,26,28
