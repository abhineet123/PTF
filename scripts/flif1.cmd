@echo off
python2 "%~dp0\..\filterLines.py" filter_strings=%1 filter_type=1 retain_filtered=%2 in_fname=%3 out_fname=%4