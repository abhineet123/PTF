@echo off
python3 "%~dp0\..\rename.py" src_substr=%1 dst_substr=%2 include_folders=%3 replace_existing=%4 show_names=%5 src_dir=%6 include_ext=1 recursive_search=1