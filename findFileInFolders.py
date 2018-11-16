import os
import sys

folder_prefix = 'Subset'
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

print 'Looking for {:s} in folders beginning with {:s} and IDs going from {:d} to {:d}'.format(
    search_str, folder_prefix, folder_start_id, folder_end_id)

total_files_found = 0
total_files_searched = 0
for folder_id in xrange(folder_start_id, folder_end_id + 1):
    src_folder = '{:s} {:d}'.format(folder_prefix, folder_id)
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





