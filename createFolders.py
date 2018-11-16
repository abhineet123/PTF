import os
import sys

filename = 'list.txt'
folder_root_dir = '.'

arg_id = 1
if len(sys.argv) > arg_id:
    filename = sys.argv[arg_id]
    arg_id += 1

if not os.path.isfile(filename):
    print 'File containing the folder list not found'
    sys.exit()

if not os.path.exists(folder_root_dir):
    os.mkdir(folder_root_dir)

data_file = open(filename, 'r')
lines = data_file.readlines()
data_file.close()

for folder_name in lines:
    folder_name = folder_name.strip()
    if len(folder_name) <= 1:
        print 'Skipping: ', folder_name
        continue
    folder_path = '{:s}/{:s}'.format(folder_root_dir, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

