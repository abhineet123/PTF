import paramparse
import os
import shutil
from pprint import pprint

from Misc import sortKey, add_suffix


class Params:

    def __init__(self):
        self.cfg = ()
        self.disable_rename = 0
        self.dst_path = '.'
        self.exceptions = []
        self.file_ext = ''
        self.folder_name = '.'
        self.include_folders = 0
        self.out_file = 'mfsf_log.txt'
        self.prefix = ''
        self.rename_to_subfolder = 0
        self.subfolder_to_prefix = 1


def main():
    params = Params()
    paramparse.process(params)

    dst_path = params.dst_path
    file_ext = params.file_ext
    out_file = params.out_file
    folder_name = params.folder_name
    prefix = params.prefix
    include_folders = params.include_folders
    disable_rename = params.disable_rename
    rename_to_subfolder = params.rename_to_subfolder
    subfolder_to_prefix = params.subfolder_to_prefix
    exceptions = params.exceptions

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
        src_files = []
        if include_folders != 2:
            src_files += [f for f in os.listdir(subfolders_path) if os.path.isfile(os.path.join(subfolders_path, f))]
            if file_ext:
                src_files = [f for f in src_files if f.endswith(file_ext)]
        if include_folders:
            src_files += [f for f in os.listdir(subfolders_path) if os.path.isdir(os.path.join(subfolders_path, f))]

        if exceptions:
            src_files = [f for f in src_files if not any([k in f for k in exceptions])]

        src_files.sort(key=sortKey)
        n_files = len(src_files)

        if n_files == 0:
            continue

        if disable_rename:
            dst_files = [f'{f}' for f in src_files]
        elif rename_to_subfolder:
            if n_files > 1:
                print(f"skipping subfolder with multiple files: {subfolder}: {src_files}")
                continue
            file_ext = os.path.splitext(src_files[0])[1]
            dst_files = [f'{subfolder}{file_ext}', ]
        elif subfolder_to_prefix:
            dst_files = [f'{subfolder}_{f}' for f in src_files]
        else:
            dst_files = [add_suffix(f, subfolder) for f in src_files]

        for i in range(n_files):
            src_path = os.path.join(subfolders_path, src_files[i])
            _dst_path = os.path.join(dst_path, dst_files[i])

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

    # print('subfolders:')
    # pprint(subfolders)
    print('total_files: {}'.format(total_files))
    out_fid.close()


if __name__ == '__main__':
    main()
