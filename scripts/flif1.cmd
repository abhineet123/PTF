@echo off
python2 "%~dp0\..\filterLines.py" in_fname=%1 filter_strings=%2 filter_type=1 retain_filtered=%3 out_fname=%4