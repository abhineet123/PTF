@echo off
python "%~dp0\..\filterLines.py" filter_strings=%1 filter_type=%2 retain_filtered=%3 in_fname=%4 out_fname=%5