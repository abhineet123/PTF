import fnmatch
import os
import shutil
import sys
import inspect
import re
from datetime import datetime

from Misc import processArguments

params = {
    'src_dir': '',
    'src_substr': '',
    'dst_substr': '',
    'recursive_search': 0,
    'include_folders': 0,
    'replace_existing': 0,
    'include_ext': 0,
    'show_names': 1,
    'convert_to_lowercase': 0,

    # 2: only log
    'write_log': 1,

    'milliseconds': 0,
    're_mode': 1,
    'exclude_src': 0,
    'print_only': 0,
    'add_as_prefix': 0,
    'add_as_suffix': 0,
}

processArguments(sys.argv[1:], params)
src_dir = params['src_dir']
src_substr = params['src_substr']
recursive_search = params['recursive_search']
include_folders = params['include_folders']
replace_existing = params['replace_existing']
include_ext = params['include_ext']
exclude_src = params['exclude_src']
show_names = params['show_names']
convert_to_lowercase = params['convert_to_lowercase']
write_log = params['write_log']
re_mode = params['re_mode']
milliseconds = params['milliseconds']
add_as_prefix = params['add_as_prefix']
add_as_suffix = params['add_as_suffix']

search_all = 0

src_dir = os.path.abspath(src_dir)
if os.path.isfile(src_dir):
    src_dir = os.path.dirname(src_dir)

if convert_to_lowercase:
    print('Converting to lower case')

if exclude_src:
    recursive_search = 1

if recursive_search:
    print('Searching recursively in all sub folders')
    if exclude_src:
        print('Excluding the direct top level folder contents')
else:
    print('Searching only in the top level folder')

if include_folders == 1:
    print('Searching for folders too')
elif include_folders == 2:
    print('Searching only for folders')
else:
    print('Not searching for folders')

if include_ext:
    print('Including file extensions as well')
else:
    print('Excluding file extensions')

if write_log == 2:
    print('Only logging the required renaming operations')

script_filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(script_filename))

if write_log:
    log_dir = os.path.join(script_path, 'log')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'rrep_log.txt')
    print(('Saving log to {}'.format(log_file)))
    log_fid = open(log_file, 'w')

src_file_paths = []
src_substrs = []
# print('Looking for matches')
iter_id = 0
for root, dirnames, filenames in os.walk(src_dir):
    if re_mode:
        if include_folders:
            for dirname in dirnames:
                matches = re.findall(src_substr, dirname)
                if matches:
                    # print('{} :: {}'.format(dirname, matches))
                    src_file_paths.append(os.path.join(root, dirname))
                    src_substrs.append(matches[0])
        if include_folders != 2:
            for filename in filenames:
                matches = re.findall(src_substr, filename)
                if matches:
                    # print('{} :: {}'.format(filename, matches))
                    src_file_paths.append(os.path.join(root, filename))
                    src_substrs.append(matches[0])
    else:
        # print('{}'.format(root))
        if include_folders:
            for dirname in fnmatch.filter(dirnames, '*{:s}*'.format(src_substr)):
                src_file_paths.append(os.path.join(root, dirname))
        if include_folders != 2:
            for filename in fnmatch.filter(filenames, '*{:s}*'.format(src_substr)):
                src_file_paths.append(os.path.join(root, filename))
    if not recursive_search:
        break
    elif exclude_src and iter_id == 0:
        src_file_paths = []
    iter_id += 1

print('Found {:d} matches'.format(len(src_file_paths)))

# if re_mode:
#     sys.exit()

src_file_roots = [os.path.dirname(src_path) for src_path in src_file_paths]
rename_dict = {}

for src_id, src_path in enumerate(src_file_paths):

    if not os.path.exists(src_path):
        root_dir = os.path.dirname(src_path)
        try:
            renamed_root_dir = rename_dict[root_dir]
        except KeyError:
            raise AssertionError('src_path does not exist though its root has not been renamed: {}'.format(
                src_path))
        else:
            src_path = src_path.replace(root_dir, renamed_root_dir)

    src_dir = os.path.dirname(src_path)
    src_fname = os.path.basename(src_path)
    src_fname_no_ext, src_ext = os.path.splitext(src_fname)

    src_image_mtime_float = os.path.getmtime(src_path)

    # src_image_fid = pathlib.Path(src_image_path)

    src_image_mtime = datetime.fromtimestamp(src_image_mtime_float)

    # print('src_image_mtime: {}'.format(src_image_mtime))

    time_fmt = "%y%m%d_%H%M%S"
    if milliseconds:
        time_fmt += '_%f'

    dst_fname_no_ext = src_image_mtime.strftime(time_fmt)

    if add_as_prefix:
        dst_fname = '{:s}_{:s}{:s}'.format(dst_fname_no_ext, src_fname_no_ext, src_ext)
    elif add_as_suffix:
        dst_fname = '{:s}_{:s}{:s}'.format(src_fname_no_ext, dst_fname_no_ext, src_ext)
    else:
        dst_fname = '{:s}{:s}'.format(dst_fname_no_ext, src_ext)

    dst_path = os.path.join(src_dir, dst_fname)

    src_fname_dir = os.path.dirname(src_path)
    dst_fname_dir = os.path.dirname(dst_path)

    if src_path != dst_path:
        dst_fname_noext, _ = os.path.splitext(dst_fname)

        seq_id = 0
        while os.path.exists(dst_path):
            seq_id += 1
            dst_fname = '{:s}_{:06d}{:s}'.format(dst_fname_noext, seq_id, src_ext)
            dst_path = os.path.join(src_dir, dst_fname)

        try:
            shutil.move(src_path, dst_path)
            # pass
        except WindowsError as e:
            print('Renaming of {} failed: {}'.format(src_path, e))
        else:
            rename_dict[src_path] = dst_path
            print('{} --> {}'.format(src_fname, dst_fname))
            if write_log:
                log_fid.write('{}\t{}\n'.format(src_path, dst_path))

if write_log:
    log_fid.close()
