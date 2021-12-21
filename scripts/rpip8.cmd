@echo off
python3 "%~dp0\..\runParallelInPython.py" batch_size=8 in_fname=%1 start_id=%2 end_id=%3