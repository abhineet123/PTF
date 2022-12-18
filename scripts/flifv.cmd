@echo off
python3 "%~dp0\..\filterLines.py" "filter_strings=.mkv,.avi,.mp4,.webm" filter_type=1 retain_filtered=%1 in_fname=%2 out_fname=%3