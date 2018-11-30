@echo off
python "%~dp0\..\rename.py" src_substr=%1 dst_substr=%2 recursive_search=%3 include_folders=2 replace_existing=%4 show_names=%5 src_dir=%6