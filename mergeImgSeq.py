import os
import sys, shutil
from pprint import pformat

from Misc import processArguments, sortKey

params = {
    'dst_path': '.',
    'file_ext': '',
    'out_file': 'mis_log.txt',
    'folder_name': '.',
    'prefix': '',
    'include_folders': 0,
    'exceptions': [],
}
processArguments(sys.argv[1:], params)
dst_path = params['dst_path']
file_ext = params['file_ext']
out_file = params['out_file']
folder_name = params['folder_name']
prefix = params['prefix']
include_folders = params['include_folders']
exceptions = params['exceptions']

dst_path = os.path.abspath(dst_path)

img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff')

if file_ext:
    img_exts = [file_ext, ]

if os.path.isfile(folder_name):
    root_dir = os.path.abspath(os.getcwd())
    subfolders = [x.strip() for x in open(folder_name).readlines()]
    print('Looking for files with extension {:s} in sub folders of {:s} listed in {}'.format(
        file_ext, root_dir, folder_name))
    folder_name = root_dir
else:
    folder_name = os.path.abspath(folder_name)
    print('Looking for files with extensions in {} in sub folders of {:s}'.format(img_exts, folder_name))
    subfolders = [name for name in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, name))]
if prefix:
    print('Limiting search to only sub folders starting with {}'.format(prefix))
    subfolders = [x for x in subfolders if x.startswith(prefix)]
try:
    subfolders.sort(key=sortKey)
except:
    subfolders.sort()

if include_folders == 1:
    print('Searching for folders too')
elif include_folders == 2:
    print('Searching only for folders')
else:
    print('Not searching for folders')

if file_ext == '__n__':
    file_ext = ''

total_files = 0
out_fid = open(out_file, 'w')
files = []
empty_folders = []
for subfolder in subfolders:
    subfolders_path = os.path.join(folder_name, subfolder)

    src_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                     os.path.splitext(f.lower())[1] in img_exts]
                    for (dirpath, dirnames, filenames) in os.walk(subfolders_path, followlinks=True)]
    src_files = [item for sublist in src_file_gen for item in sublist]

    if exceptions:
        src_files = [f for f in src_files if not any([k in f for k in exceptions])]

    src_files.sort(key=sortKey)
    n_files = len(src_files)

    dst_files = ['{}_{}'.format(subfolder, f) for f in src_files]

    for i in range(n_files):
        src_path = src_files[i]
        src_filename = os.path.basename(src_path)

        if src_filename[0].isdigit():
            continue

        _dst_path = os.path.join(subfolders_path, src_filename)
        src_dir = os.path.dirname(src_path)
        if src_path != _dst_path:
            empty_folders.append(src_dir)

        src_dir_name = os.path.basename(src_dir)
        _dst_path = os.path.join(subfolders_path, '{}_{}'.format(src_dir_name, src_filename))

        print('{} -> {}'.format(src_path, _dst_path))
        try:
            shutil.move(src_path, _dst_path)
            out_fid.write('{}\t{}\n'.format(src_path, _dst_path))
        except shutil.Error as e:
            print('shutil.Error Failure: {}'.format(e))
            continue
        except OSError as e:
            print('OSError Failure: {}'.format(e))
            continue
        except BaseException as e:
            print('BaseException Failure: {}'.format(e))
            continue
        total_files += 1

empty_folders = list(set(empty_folders))
if empty_folders:
    print('Removing empty_folders:\n{}'.format(pformat(empty_folders)))
    for _folder in empty_folders:
        if os.path.isdir(_folder):
            shutil.rmtree(_folder)
# print('subfolders:')
# pprint(subfolders)
print('total_files: {}'.format(total_files))
out_fid.close()
