import os
import sys, shutil

from Misc import processArguments, sortKey

params = {
    'dst_path': '.',
    'file_ext': 'jpg',
    'out_file': 'rsfc_log.txt',
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
exceptions = params['exceptions']

dst_path = os.path.abspath(dst_path)

if os.path.isfile(folder_name):
    root_dir = os.path.abspath(os.getcwd())
    subfolders = [x.strip() for x in open(folder_name).readlines()]
    print('Looking for files with extension {:s} in sub folders of {:s} listed in {}'.format(
        file_ext, root_dir, folder_name))
    folder_name = root_dir
else:
    folder_name = os.path.abspath(folder_name)
    print('Looking for files with extension {:s} in sub folders of {:s}'.format(file_ext, folder_name))
    subfolders = [name for name in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, name))]
if prefix:
    print('Limiting search to only sub folders starting with {}'.format(prefix))
    subfolders = [x for x in subfolders if x.startswith(prefix)]
try:
    subfolders.sort(key=sortKey)
except:
    subfolders.sort()


if file_ext == '__n__':
    file_ext = ''

total_files = 0
out_fid = open(out_file, 'w')
files = []
empty_folders = []
for subfolder in subfolders:
    subfolders_path = os.path.join(folder_name, subfolder)
    src_files = []
    src_files += [f for f in os.listdir(subfolders_path) if os.path.isfile(os.path.join(subfolders_path, f))]
    if file_ext:
        src_files = [f for f in src_files if f.endswith(file_ext)]

    if exceptions:
        src_files = [f for f in src_files if not any([k in f for k in exceptions])]

    src_files.sort(key=sortKey)
    n_files = len(src_files)

    assert n_files > 1, "only one file found in {}".format(subfolders_path)

    first_file, last_file = src_files[0], src_files[-1]

    first_file_no_ext, _ = os.path.splitext(first_file)
    last_file_no_ext, _ = os.path.splitext(last_file)

    dst_subfolders_name = '{}_to_{}'.format(first_file_no_ext, last_file_no_ext)

    if dst_subfolders_name == subfolder:
        continue

    dst_subfolders_path = os.path.join(folder_name, dst_subfolders_name)

    print('{} -> {}'.format(subfolders_path, dst_subfolders_path))


    try:
        shutil.move(subfolders_path, dst_subfolders_path)
        out_fid.write('{}\t{}\n'.format(subfolders_path, dst_subfolders_path))
    except shutil.Error as e:
        print('shutil.Error Failure: {}'.format(e))
        continue
    except OSError as e:
        print('OSError Failure: {}'.format(e))
        continue
    except BaseException as e:
        print('BaseException Failure: {}'.format(e))
        continue

out_fid.close()
