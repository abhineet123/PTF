import os
import sys
from pprint import pformat
from Misc import sortKey

root_dir = '.'
folder_prefix = ''
folder_start_id = 1
folder_end_id = 100
search_str = ''
find_unique_names = 1

arg_id = 1
if len(sys.argv) > arg_id:
    search_str = sys.argv[arg_id]
    arg_id += 1
if len(sys.argv) > arg_id:
    folder_end_id = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    folder_start_id = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    folder_prefix = sys.argv[arg_id]
    arg_id += 1
if len(sys.argv) > arg_id:
    find_unique_names = int(sys.argv[arg_id])
    arg_id += 1

exclusions = ['Thumbs.db', 'sync.ffs_db']
img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff')

if folder_prefix:
    print('Looking for {:s} in folders beginning with {:s} and IDs going from {:d} to {:d}'.format(
        search_str, folder_prefix, folder_start_id, folder_end_id))
    src_folders = ['{:s} {:d}'.format(folder_prefix, folder_id) for folder_id in
                   range(folder_start_id, folder_end_id + 1)]
else:
    src_folders = [os.path.join(root_dir, k) for k in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, k))]

src_folders = sorted(src_folders, key=sortKey)

total_files_found = 0
total_files_searched = 0
total_unique_names = 0
all_unique_names = []
matching_files = {}
for src_folder in src_folders:
    if not os.path.exists(src_folder):
        break
    src_files = [f for f in os.listdir(src_folder) if f not in exclusions and
                 os.path.isfile(os.path.join(src_folder, f))]

    n_src_files = len(src_files)
    search_results = [f for f in src_files if search_str in f]
    if len(search_results) > 0:
        print('Found {:d} matching files in {:s}'.format(len(search_results), src_folder))
        print('\n'.join(search_results) + '\n')
        total_files_found += len(search_results)

        matching_files[src_folder] = search_results
    else:
        print('Done searching {:s} with {:d} files'.format(src_folder, n_src_files))

    unique_names = []
    if find_unique_names:
        src_files = [os.path.splitext(f)[0] for f in src_files if os.path.splitext(f)[1] in img_exts]
        if src_files:
            unique_names.append(src_files[0])
        for i in range(1, len(src_files)):
            commonprefix = os.path.commonprefix(src_files[i - 1:i + 1])
            if not commonprefix:
                unique_names.append(src_files[i])
                continue

            non_prefix = src_files[i].replace(commonprefix, '')
            found_digit = 0
            for _c in non_prefix:
                if str(_c).isdigit():
                    found_digit = 1
                    break
                if str(_c).isalpha():
                    unique_names.append(src_files[i])
                    break
    if unique_names:
        print('Found {} unique names:\n{}\n'.format(len(unique_names), unique_names))
        total_unique_names += len(unique_names)
        all_unique_names += unique_names

    total_files_searched += len(src_files)


def extract_name(_str):
    _str_list = _str.split('_')
    _names = []
    for i, _substr in enumerate(_str_list):
        if not all(k.isalpha() for k in _substr):
            break
        _names.append(_substr)
    _name = ' '.join(_names)
    return _name


print('{:d} files searched'.format(total_files_searched))
all_unique_names.sort()

all_unique_names_proc = list(map(extract_name, all_unique_names))
all_unique_names_cmb = list(map(lambda x: '\t'.join(x), zip(all_unique_names, all_unique_names_proc)))

if find_unique_names:
    print('{:d} unique names found:\n{}'.format(total_unique_names, '\n'.join(all_unique_names_proc)))

if total_files_found > 0:
    print('{} matching files found:\n{}'.format(total_files_found, pformat(matching_files)))
else:
    print('No matching files found')
