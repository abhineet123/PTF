@echo off
python3 "%~dp0\..\findFileInFolders.py" collage=1 recursive=1 search_str=%1 folder_end_id=%2 folder_start_id=%3 folder_prefix=%4 excluded="vids,collage_old,0,dummy,#patches_src,#unsorted,#bad,#semi_bad"
