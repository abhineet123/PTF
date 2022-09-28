@echo off
python3 "%~dp0\..\rename.py" include_folders=2 recursive_search=1 src_substr=%1 dst_substr=%2 convert_to_lowercase=1 replace_existing=%4 show_names=%5 src_dir=%6