@echo off
python "%~dp0\..\rename.py" src_substr=%1 dst_substr=%2 recursive_search=%3 include_folders=%4 replace_existing=%5 show_names=%6 src_dir=%7 include_ext=1