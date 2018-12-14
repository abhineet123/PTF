@echo off
python "%~dp0\..\printDirectoryTree.py" markdown_mode=1 exts_to_include=%1 strings_to_exclude=%2 fix_weird_text=%3