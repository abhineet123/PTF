@echo off
python3 "%~dp0\..\visualizeWithMotion.py" on_top=0 top_border=0 keep_borders=1 n_images=1 random_mode=1 auto_progress=1 transition_interval=60 monitor_id=0  duplicate_window=0 second_from_top=1 win_offset_y=0 width=1920 height=1050 src_dirs=!!#,!!##,!!#bad,!!#proc,!!#seg_masks,!!#masks,!!#contours,24/1*2,24/2*6,24*8,25*6,26*4,27*2,18*2,19*2,21*2,28*2,29*3,17/2*2,2__to__16,30,31,32 frg_win_titles="Visual Studio Code" frg_monitor_ids=0,1,2,4,5 only_maximized=0 reversed_pos=2 min_aspect_ratio=0 auto_min_aspect_ratio=0 max_aspect_ratio=1.8 auto_aspect_ratio=1 max_magnified_height_ratio=2 save_magnified=1 src_dirs=!!#,!!##,!!#bad,!!#proc,!!#seg_masks,!!#masks,!!#contours,!!collage,%1 id_probs=id_probs.csv n_images=1
REM target_aspect_ratio=0.2
REM n_images=12 custom_grid_size=6x2
REM min_aspect_ratio=0.3 auto_min_aspect_ratio=0.25
REM n_images=10 custom_grid_size=5x2
REM id_probs=id_probs.csv
REM "x99","grs","orca","x992","grs2","orca2","f","t"
REM frg_win_titles="The Journal 8","!!!XYplorer 20.10"
REM ,"PyCharm"
REM ,"!dummy.log - Visual Studio Code"
REM ,"!Visual Studio Code"
REM ,landscape*20
REM vids/20/2/13**20,vids/20/2_patches**20
REM ,20/1/1_4,20/1/1_5,20/2,17/2,17/3//2,17/4//2
REM 24/4*2
REM 0/1,0/2,0/3,0/4,0/5,0/6,0/7,0/8,0/10,0/11

REM 24/1*2,24/2*6,24*8,25*6,26*4,27*2,18*2,19,21*2,28,29*3,30,17/2*2,17/3//3,17/4///2,0,2__to__16,31,32,20/1_1///2,20/1_4,20/1_5,20/2,20/1_3///3,1
REM ,17/3//4,17/4///2,0,20/1_1,20/1_4,20/1_5,20/2,20/1_3///4,1
