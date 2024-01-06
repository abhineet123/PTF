@echo off
python3 "%~dp0\..\rename.py" re_mode=1 src_substr=%1 dst_substr=%2 recursive_search=1 include_folders=%4 replace_existing=%5 show_names=%6 src_dir=%7 include_ext=1