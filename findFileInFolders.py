import os
import shutil
import sys
from pprint import pformat
from pathlib import Path

import cv2

from Misc import sortKey, processArguments, stackImages

if __name__ == '__main__':
    params = {
        'root_dir': '.',
        'folder_prefix': '',
        'folder_start_id': 1,
        'folder_end_id': 100,
        'search_str': '',
        'find_unique_names': 1,
        'recursive': 0,
        'collage': 0,
    }
    processArguments(sys.argv[1:], params)
    root_dir = params['root_dir']
    folder_prefix = params['folder_prefix']
    folder_start_id = params['folder_start_id']
    folder_end_id = params['folder_end_id']
    search_str = params['search_str']
    find_unique_names = params['find_unique_names']
    recursive = params['recursive']
    collage = params['collage']

    # arg_id = 1
    # if len(sys.argv) > arg_id:
    #     search_str = sys.argv[arg_id]
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     folder_end_id = int(sys.argv[arg_id])
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     folder_start_id = int(sys.argv[arg_id])
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     folder_prefix = sys.argv[arg_id]
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     find_unique_names = int(sys.argv[arg_id])
    #     arg_id += 1

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
    all_collage_images = []
    matching_files = {}
    n_src_files = {}

    if src_folders:
        collage_path = os.path.join(os.path.dirname(src_folders[0]), 'collage')
        collage_path = os.path.abspath(collage_path)
        if not os.path.isdir(collage_path):
            os.makedirs(collage_path)

    for src_folder in src_folders:
        src_folder = os.path.abspath(src_folder)

        if src_folder == collage_path:
            continue

        src_folder_name = os.path.basename(src_folder)

        if not os.path.exists(src_folder):
            print('src_folder does not exist: {}'.format(src_folder))
            break

        if recursive:
            src_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                             os.path.splitext(f.lower())[1] in img_exts]
                            for (dirpath, dirnames, filenames) in os.walk(src_folder, followlinks=True)]
            src_paths = [item for sublist in src_file_gen for item in sublist]
        else:
            src_paths = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f not in exclusions and
                         os.path.isfile(os.path.join(src_folder, f)) and os.path.splitext(f.lower())[1] in img_exts]

        src_files = [os.path.basename(f) for f in src_paths]
        n_src_files[src_folder] = len(src_files)
        search_results = [f for f in src_files if search_str in f]
        if len(search_results) > 0:
            print('Found {:d} matching files in {:s}'.format(len(search_results), src_folder))
            print('\n'.join(search_results) + '\n')
            total_files_found += len(search_results)

            matching_files[src_folder] = search_results
        else:
            print('Done searching {:s} with {:d} files'.format(src_folder, n_src_files[src_folder]))

        unique_names = []
        collage_images = []
        if find_unique_names:
            src_files_no_ext = [os.path.splitext(f)[0] for f in src_files]
            if src_files_no_ext:
                unique_names.append(src_files_no_ext[0])
                if collage:
                    # collage_images.append(cv2.imread(src_paths[0]))
                    _name, _ext = os.path.splitext(src_files[0])
                    dst_file = '{}_{}{}'.format(_name, src_folder_name, _ext)
                    shutil.copy(src_paths[0], os.path.join(collage_path, dst_file))
            for i in range(1, len(src_files_no_ext)):
                commonprefix = os.path.commonprefix(src_files_no_ext[i - 1:i + 1])
                if not commonprefix:
                    unique_names.append(src_files_no_ext[i])
                    if collage:
                        # collage_images.append(cv2.imread(src_paths[i]))
                        _name, _ext = os.path.splitext(src_files[i])
                        dst_file = '{}_{}{}'.format(_name, src_folder_name, _ext)
                        shutil.copy(src_paths[i], os.path.join(collage_path, dst_file))

                    continue

                non_prefix = src_files_no_ext[i].replace(commonprefix, '')
                found_digit = 0
                for _c in non_prefix:
                    if str(_c).isdigit():
                        found_digit = 1
                        break
                    if str(_c).isalpha():
                        unique_names.append(src_files_no_ext[i])
                        if collage:
                            # collage_images.append(cv2.imread(src_paths[i]))
                            _name, _ext = os.path.splitext(src_files[i])
                            dst_file = '{}_{}{}'.format(_name, src_folder_name, _ext)
                            shutil.copy(src_paths[i], os.path.join(collage_path, dst_file))
                        break
        if unique_names:
            print('Found {} unique names:\n{}\n'.format(len(unique_names), unique_names))
            total_unique_names += len(unique_names)
            all_unique_names += unique_names
            # if collage:
            #     all_collage_images += collage_images

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

    # all_unique_names_proc = list(map(extract_name, all_unique_names))
    all_unique_names_proc = [x for x in map(extract_name, all_unique_names) if x.strip()]

    # all_unique_names_cmb = list(map(lambda x: '\t'.join(x), zip(all_unique_names, all_unique_names_proc)))

    if find_unique_names:
        print('{:d} unique names found:\n{}'.format(total_unique_names, '\n'.join(all_unique_names_proc)))

    if total_files_found > 0:
        print('\n{} matching files found:\n{}'.format(total_files_found, pformat(matching_files)))
    else:
        print('\nNo matching files found')

    n_src_files = [(k, v) for k, v in sorted(n_src_files.items(), key=lambda item: item[1])]
    print('\nn_src_files:\n{}'.format(pformat(n_src_files)))

    # if collage and all_collage_images:
    #     collage_path = os.path.join('.', 'collage.jpg')
    #     print('\ncreating collage image from {} images...'.format(len(all_collage_images)))
    #     collage_img = stackImages(all_collage_images)
    #     print('\nsaving collage image to: {}'.format(collage_path))
    #
    #     cv2.imwrite(collage_path, collage_img)


