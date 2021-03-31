@echo off
python3 "%~dp0\..\visualizeWithMotion.py" on_top=0 top_border=0 keep_borders=1 n_images=1 random_mode=1 auto_progress=1 transition_interval=15 monitor_id=0  duplicate_window=0 second_from_top=1 win_offset_y=0 width=1920 height=1050 src_dirs=!!#,!!##,24/1*4,24/2*6,24*8,25*4,26*2,27 frg_win_titles="The Journal 8","!dummy.log - Visual Studio Code" frg_monitor_ids=0,1,2,4,5 only_maximized=0 reversed_pos=2 src_path=%1 n_images=%2

REM "x99","grs","orca","x992","grs2","orca2","f","t"
REM frg_win_titles="The Journal 8","!!!XYplorer 20.10"
REM ,"PyCharm"
REM ,"!dummy.log - Visual Studio Code"
REM ,"!Visual Studio Code"

REM vids/20/2/13**20,vids/20/2_patches**20

