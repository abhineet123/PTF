python videoToImgSeq.py db_root_dir=H:/UofA/Acamp/code/object_detection/videos actor=2018_05_29_CalgaryZoo\Bear seq_name=M2U00055 vid_fmt=MPG

python videoToImgSeq.py db_root_dir=H:/UofA/Acamp/code/object_detection/videos actor=. seq_name=human3 vid_fmt=mkv

python3 videoToImgSeq.py db_root_dir=/home/abhineet/acamp/acamp_code/object_detection/videos actor=. seq_name=human3 vid_fmt=mp4
python3 videoToImgSeq.py db_root_dir=/home/abhineet/acamp/acamp_code/object_detection/videos actor=. seq_name=human4 vid_fmt=mkv
python3 videoToImgSeq.py db_root_dir=/home/abhineet/acamp/acamp_code/object_detection/videos actor=. seq_name=human5 vid_fmt=mkv

python renameFilesIntoSeq.py image 1 0 1 H:/UofA/Acamp/code/object_detection/videos/2018_05_29_CalgaryZoo/Bear/M2U00055

python renameFilesIntoSeq.py image 1 0 1 /home/abhineet/H/UofA/Acamp/code/object_detection/videos/2018_05_29_CalgaryZoo/Bear/M2U00055

ssh-keygen -t rsa -C "asingh@acamp.ca" -b 4096