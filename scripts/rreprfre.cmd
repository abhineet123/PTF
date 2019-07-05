@echo off
python2 "%~dp0\..\rename.py" re_mode=1 recursive_search=1 include_folders=1 src_substr=%1 dst_substr=%2 convert_to_lowercase=%3 replace_existing=%4 show_names=%5 src_dir=%6