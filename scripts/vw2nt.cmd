@echo off
python3 "%~dp0\..\visualizeWithMotion.py" src_dirs=17/2,17/3,17/4,17/5,17/7,4,18,19,20,0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16 fullscreen=1 random_mode=1 auto_progress=1 trim_images=1 n_images=6 n_images=%1 transition_interval=15  transition_interval=%2 on_top=0

