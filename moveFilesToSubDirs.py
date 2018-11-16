import sys
import os
import random
import shutil

subdir_prefix = 'Set'
subdir_root_dir = '.'
files_per_subdir = 100
subdir_start_id = 1
rename_files = 1
shuffle_files = 1

arg_id = 1
if len(sys.argv) > arg_id:
    subdir_prefix = sys.argv[arg_id]
    arg_id += 1
if len(sys.argv) > arg_id:
    files_per_subdir = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    subdir_start_id = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    rename_files = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    shuffle_files = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    subdir_root_dir = sys.argv[arg_id]
    arg_id += 1

print 'subdir_prefix: {:s}'.format(subdir_prefix)
print 'files_per_subdir: {:d}'.format(files_per_subdir)
print 'subdir_start_id: {:d}'.format(subdir_start_id)


src_file_names = [f for f in os.listdir(subdir_root_dir) if os.path.isfile(os.path.join(subdir_root_dir, f))]
if shuffle_files:
    print 'Shuffling files...'
    random.shuffle(src_file_names)

subdir_id = subdir_start_id
file_count = 0
file_id = 1
for src_fname in src_file_names:
    filename, file_extension = os.path.splitext(src_fname)
    src_path = os.path.join(subdir_root_dir, src_fname)
    if rename_files:
        dst_fname = '{:s}{:d}_{:d}.{:s}'.format(subdir_prefix, subdir_id, file_id, file_extension)
    else:
        dst_fname = src_fname
    dst_subdir_name = '{:s} {:d}'.format(subdir_prefix, subdir_id)
    dst_dir = os.path.join(subdir_root_dir, dst_subdir_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst_path = os.path.join(dst_dir, dst_fname)
    shutil.move(src_path, dst_path)
    if file_id == files_per_subdir:
        print 'Done sub dir {:d}'.format(subdir_id)
        subdir_id += 1
        file_id = 1
    else:
        file_id += 1
