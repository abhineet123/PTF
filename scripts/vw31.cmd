@echo off
python3 "%~dp0\..\visualizeWithMotion.py" src_dirs=20/1 fullscreen=0 random_mode=1 auto_progress=1 trim_images=1 n_images=2 n_images=%1 transition_interval=10000  transition_interval=%2 enable_hotkeys=1 show_window=0 reversed_pos=0 custom_grid_size=0x1 mode=1 borderless=0

