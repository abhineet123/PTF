@echo off
python3 "%~dp0\..\filterLines.py" in_fname=%1 filter_strings=%2 filter_type=%3 retain_filtered=%4  out_fname=%5