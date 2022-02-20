@echo off
python3 "%~dp0\..\findSimilarImages.py" files=%1 root_dir=%2 feature_type=1 db_file=fsi_vgg_db.pkl