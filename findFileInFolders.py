import os
import sys

root_dir = '.'
folder_prefix = ''
folder_start_id = 1
folder_end_id = 100
search_str = ''

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

if folder_prefix:
    print 'Looking for {:s} in folders beginning with {:s} and IDs going from {:d} to {:d}'.format(
        search_str, folder_prefix, folder_start_id, folder_end_id)
    src_folders = ['{:s} {:d}'.format(folder_prefix, folder_id) for folder_id in
                   xrange(folder_start_id, folder_end_id + 1)]
else:
    src_folders = [os.path.join(root_dir, k) for k in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, k))]

total_files_found = 0
total_files_searched = 0
for src_folder in src_folders:
    if not os.path.exists(src_folder):
        break
    src_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    search_results = [f for f in src_files if search_str in f]
    if len(search_results) > 0:
        print 'Found {:d} matching files in {:s}'.format(len(search_results), src_folder)
        print '\n'.join(search_results)
        total_files_found += len(search_results)
    else:
        print 'Done searching {:s}'.format(src_folder)
    total_files_searched += len(src_files)
if total_files_found > 0:
    print 'Total matching files found : {:d}'.format(total_files_found)
else:
    print 'No matching files found'
print '{:d} files searched'.format(total_files_searched)
