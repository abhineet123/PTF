@echo off
python3 "%~dp0\..\splitImgSeq.py" thresh=-1 src_path=%1 metric=%2 order=%3
