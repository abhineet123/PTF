@echo off
python3 "%~dp0\..\rename.py" src_substr=" - Shortcut" dst_substr=%2 recursive_search=%3 include_folders=%4 convert_to_lowercase=%5 replace_existing=%6 show_names=%7 src_dir=%8