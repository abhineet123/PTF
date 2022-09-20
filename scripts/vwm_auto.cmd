@echo off
python3 "%~dp0\..\visualizeWithMotion.py" src_dirs=!!#,!!##,!!#bad,!!#proc,!!#seg_masks,!!#masks,!!#contours,!!collage,,24/1*2,24/2*6,24*8,25*6,26*4,27*2,18*2,19*2,21*2,28*2,29*3,17/2*2,20 n_images=%2 lazy_video_load=0 fullscreen=1 exclude_src_dirs="masks,contours" trim_images=0 random_mode=1 auto_progress=1 transition_interval=15

