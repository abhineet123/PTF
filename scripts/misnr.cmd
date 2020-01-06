@echo off
python3 "%~dp0\..\mergeImgSeq.py" rename=0 folder_name=%1 prefix=%2 exceptions=%3 file_ext=%4
