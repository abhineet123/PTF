import os
import sys, shutil
from random import shuffle

from Misc import processArguments, sortKey, linux_path

params = {
    'file_ext': '',
    'out_file': 'list.txt',
    'folder_name': '.',
    'prefix': '',
    'shuffle_files': 1,
    'del_empty': 0,
    'recursive': 1,
    'sort_by_count': 0,
}
processArguments(sys.argv[1:], params)
file_ext = params['file_ext']
out_file = params['out_file']
folder_name = params['folder_name']
shuffle_files = params['shuffle_files']
del_empty = params['del_empty']
prefix = params['prefix']
recursive = params['recursive']
sort_by_count = params['sort_by_count']

if os.path.isfile(folder_name):
    root_dir = os.path.abspath(os.getcwd())
    subfolders = [x.strip() for x in open(folder_name).readlines()]
    print('Looking for files with extension {:s} in sub folders of {:s} listed in {}'.format(
        file_ext, root_dir, folder_name))
    folder_name = root_dir
    subfolders = [linux_path(folder_name, name) for name in subfolders]
else:
    folder_name = os.path.abspath(folder_name)
    print('Looking for files with extension {:s} in sub folders of {:s}'.format(file_ext, folder_name))
    if recursive:
        print('Searching recursively')
        subfolders_gen = [[dirpath]
                          for (dirpath, dirnames, filenames) in os.walk(folder_name, followlinks=True)]
        subfolders = [item for sublist in subfolders_gen for item in sublist if item != folder_name]
    else:
        subfolders = [linux_path(folder_name, name) for name in os.listdir(folder_name) if
                      os.path.isdir(linux_path(folder_name, name))]
if prefix:
    print('Limiting search to only sub folders starting with {}'.format(prefix))
    subfolders = [x for x in subfolders if os.path.basename(x).startswith(prefix)]

subfolders.sort(key=lambda x: os.path.relpath(x, folder_name))

seq_info_file = open('seq_info.txt', 'w')
counts_file = open('file_counts.txt', 'w')

if sort_by_count:
    print('Sorting list by counts')

n_files_list = []
src_files_list = []
for subfolders_path in subfolders:
    src_files = [linux_path(f) for f in os.listdir(subfolders_path) if os.path.isfile(linux_path(subfolders_path, f))]
    if file_ext:
        src_files = [f for f in src_files if f.endswith(file_ext)]
    src_files.sort(key=sortKey)

    n_files = len(src_files)
    src_files_list.append(src_files)
    n_files_list.append(n_files)

n_seq = len(n_files_list)
print('n_seq: {}'.format(n_seq))

sort_idx = range(n_seq)
if sort_by_count:
    sort_idx = sorted(sort_idx, key=lambda k: n_files_list[k])

total_files = 0
files = []
empty_folders = []
out_text = ''
n_non_empty = 0
for i, _idx in enumerate(sort_idx):
    subfolders_path = linux_path(subfolders[_idx])
    src_files = src_files_list[_idx]
    n_files = n_files_list[_idx]
    total_files += n_files

    if n_files == 0:
        empty_folders.append(subfolders_path)
        print(f'empty_folder: {subfolders_path}')
    else:
        n_non_empty += 1

        files += [linux_path(subfolders_path, f) for f in src_files]
        counts_text = f'{n_non_empty}\t{subfolders_path}\t{n_files}\t{total_files}'
        print(counts_text)
        counts_file.write(counts_text + '\n')

        # subfolders_name = os.path.basename(subfolders_path)
        subfolders_name = linux_path(os.path.relpath(subfolders_path, folder_name))
        seq_info_text = f"{n_non_empty - 1}: ('{subfolders_name}', {n_files}),"
        seq_info_file.write(seq_info_text + '\n')

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
seq_info_file.close()

if shuffle_files:
    shuffle(files)

out_fid = open(out_file, 'w')
for f in files:
    out_fid.write(f + '\n')
out_fid.close()
