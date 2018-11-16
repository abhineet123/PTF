import os
import sys

src_folder = '.'
search_str = ''
search_sub_dir = 0

arg_id = 1
if len(sys.argv) > arg_id:
    search_str = sys.argv[arg_id]
    arg_id += 1
if len(sys.argv) > arg_id:
    search_sub_dir = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    src_folder = sys.argv[arg_id]
    arg_id += 1

print 'Looking for {:s} in {:s}'.format(search_str, src_folder)
if search_sub_dir:
    print 'Looking in sub directories too'

total_files_found = 0
if search_sub_dir:
    src_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    for root, subdirs, files in os.walk(src_folder):
        print subdirs
        for subdir in subdirs:
            src_files.append([f for f in os.listdir(os.path.join(src_folder, subdir)) if os.path.isfile(os.path.join(src_folder, subdir, f))])
else:
    src_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]



search_results = [f for f in src_files if search_str in f]
if len(search_results) > 0:
    print 'Found {:d} matching files'.format(len(search_results))
    print '\n'.join(search_results)
    total_files_found += len(search_results)
if total_files_found > 0:
    print 'Total matching files found : {:d}'.format(total_files_found)
else:
    print 'No matching files found'
print '{:d} files searched'.format(len(src_files))





