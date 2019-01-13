@echo off
python "%~dp0\..\printDirectoryTree.py" start_path=%1 markdown_mode=1 exts_to_include=pdf strings_to_exclude=.git fix_weird_text=1