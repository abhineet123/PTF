@echo off
python2 "%~dp0\..\rename.py" re_mode=1 include_folders=2 src_substr=%1 dst_substr=%2 recursive_search=%3 convert_to_lowercase=%4 replace_existing=%5 show_names=%6 src_dir=%7