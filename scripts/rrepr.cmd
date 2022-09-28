@echo off
python3 "%~dp0\..\rename.py" recursive_search=1 src_substr=%1 dst_substr=%2 include_folders=%3 convert_to_lowercase=%4 replace_existing=%5 show_names=%6 src_dir=%7