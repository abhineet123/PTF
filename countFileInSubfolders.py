import os
import sys, shutil
from random import shuffle

from Misc import processArguments, sortKey

params = {
    'file_ext': 'jpg',
    'out_file': 'list.txt',
    'folder_name': '.',
    'prefix': '',
    'shuffle_files': 1,
    'del_empty': 0,
    'recursive': 1,
}
processArguments(sys.argv[1:], params)
file_ext = params['file_ext']
out_file = params['out_file']
folder_name = params['folder_name']
shuffle_files = params['shuffle_files']
del_empty = params['del_empty']
prefix = params['prefix']
recursive = params['recursive']

if os.path.isfile(folder_name):
    root_dir = os.path.abspath(os.getcwd())
    subfolders = [x.strip() for x in open(folder_name).readlines()]
    print('Looking for files with extension {:s} in sub folders of {:s} listed in {}'.format(
        file_ext, root_dir, folder_name))
    folder_name = root_dir
    subfolders = [os.path.join(folder_name, name) for name in subfolders]
else:
    folder_name = os.path.abspath(folder_name)
    print('Looking for files with extension {:s} in sub folders of {:s}'.format(file_ext, folder_name))
    if recursive:
        print('Searching recursively')
        subfolders_gen = [[dirpath]
                          for (dirpath, dirnames, filenames) in os.walk(folder_name, followlinks=True)]
        subfolders = [item for sublist in subfolders_gen for item in sublist]
    else:
        subfolders = [os.path.join(folder_name, name) for name in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, name))]
if prefix:
    print('Limiting search to only sub folders starting with {}'.format(prefix))
    subfolders = [x for x in subfolders if os.path.basename(x).startswith(prefix)]
try:
    subfolders.sort(key=sortKey)
except:
    subfolders.sort()

total_files = 0
counts_file = open('file_counts.txt', 'w')
files = []
empty_folders = []
for subfolders_path in subfolders:
    # subfolders_path = os.path.join(folder_name, subfolder)
    subfolder = os.path.relpath(folder_name, subfolders_path)
    src_files = [f for f in os.listdir(subfolders_path) if os.path.isfile(os.path.join(subfolders_path, f))]
    if file_ext:
        src_files = [f for f in src_files if f.endswith(file_ext)]
    src_files.sort(key=sortKey)

    n_files = len(src_files)

    if n_files == 0:
        empty_folders.append(subfolders_path)
    else:
        total_files += n_files
        files += [os.path.join(subfolders_path, f) for f in src_files]
        text = '{}:\t{}\t{}'.format(subfolders_path, n_files, total_files)
        print(text)
        counts_file.write(text + '\n')

print('total_files: {}'.format(total_files))
if empty_folders:
    if del_empty:
        print('Removing empty folders:')
        for folder in empty_folders:
            shutil.rmtree(folder)
            print(folder)
    else:
        print('empty_folders:\n{}'.format('\n'.join(empty_folders)))

counts_file.close()

if shuffle_files:
    shuffle(files)

out_fid = open(out_file, 'w')
for f in files:
    out_fid.write(f + '\n')
out_fid.close()


