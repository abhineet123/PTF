@echo off
python "%~dp0\..\videoToImgSeq.py" seq_name=%1 reverse=1 ext=png n_frames=%2 start_id=%3 dst_dir=%4