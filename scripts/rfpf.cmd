@echo off
python3 "%~dp0\..\rename.py" include_folders=2 src_substr=__pf__ dst_substr=__rf__ recursive_search=%3 convert_to_lowercase=%4 replace_existing=%5 show_names=%6 src_dir=%7