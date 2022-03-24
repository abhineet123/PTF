import os
import random
import shutil

import paramparse

params = {
    'root_dir': '.',
    'subdir_name_from_files': 0,
    'subdir_prefix': '',
    'files_per_subdir': 100,
    'subdir_start_id': 1,
    'rename_files': 0,
    'shuffle_files': 0,
}
paramparse.process_dict(params)

root_dir = params['root_dir']
subdir_name_from_files = params['subdir_name_from_files']
subdir_prefix = params['subdir_prefix']
files_per_subdir = params['files_per_subdir']
subdir_start_id = params['subdir_start_id']
rename_files = params['rename_files']
shuffle_files = params['shuffle_files']

print('subdir_prefix: {:s}'.format(subdir_prefix))
print('files_per_subdir: {:d}'.format(files_per_subdir))
print('subdir_start_id: {:d}'.format(subdir_start_id))

src_file_names = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
if shuffle_files:
    print('Shuffling files...')
    random.shuffle(src_file_names)

subdir_id = subdir_start_id
file_count = 0
dir_file_id = 1

n_files = len(src_file_names)

first_file_name = None
for src_id, src_fname in enumerate(src_file_names):
    filename, file_extension = os.path.splitext(src_fname)
    src_path = os.path.join(root_dir, src_fname)

    if subdir_name_from_files or subdir_prefix:
        if rename_files:
            dst_fname = '{:s}{:d}_{:d}.{:s}'.format(subdir_prefix, subdir_id, dir_file_id, file_extension)
        else:
            dst_fname = src_fname
        dst_subdir_name = '{:s}{:d}'.format(subdir_prefix, subdir_id)
    else:
        dst_fname = src_fname
        dst_subdir_name = os.path.splitext(src_fname)[0]

    dst_dir = os.path.join(root_dir, dst_subdir_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_fname_no_ext = os.path.splitext(dst_fname)[0]
    if first_file_name is None:
        first_file_name = dst_fname_no_ext

    dst_path = os.path.join(dst_dir, dst_fname)
    shutil.move(src_path, dst_path)
    if dir_file_id == files_per_subdir or src_id == n_files - 1:
        print('Done sub dir {:d}'.format(subdir_id))

        if subdir_name_from_files:
            final_dst_subdir_name = '{}_to_{}'.format(first_file_name, dst_fname_no_ext)
            print('changing name to {}'.format(final_dst_subdir_name))

            final_dst_dir = os.path.join(root_dir, final_dst_subdir_name)

            shutil.move(dst_dir, final_dst_dir)

        subdir_id += 1
        first_file_name = None
        dir_file_id = 1
    else:
        dir_file_id += 1
