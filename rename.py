import fnmatch
import os
import shutil
import sys
import inspect
import re

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

    're_mode': 1,
    'exclude_src': 0,
    'print_only': 0,
}

processArguments(sys.argv[1:], params)
src_dir = params['src_dir']
src_substr = params['src_substr']
dst_substr = params['dst_substr']
recursive_search = params['recursive_search']
include_folders = params['include_folders']
replace_existing = params['replace_existing']
include_ext = params['include_ext']
exclude_src = params['exclude_src']
show_names = params['show_names']
convert_to_lowercase = params['convert_to_lowercase']
write_log = params['write_log']
re_mode = params['re_mode']

add_as_prefix = 0
add_as_suffix = 0
remove_files = 0
rename_files = 0
search_all = 0

src_dir = os.path.abspath(src_dir)
if os.path.isfile(src_dir):
    src_dir = os.path.dirname(src_dir)

dst_substr_orig = dst_substr

if src_substr == '__all__' or src_substr == '__a__':
    src_substr = ''
elif src_substr == '__clipboard__' or src_substr == '__cb__':
    if not src_substr:
        print('Getting src_substr from clipboard')
        from tkinter import Tk

        src_substr = Tk().clipboard_get()
if src_substr == '__prefix__' or src_substr == '__pf__':
    src_substr = ''
    add_as_prefix = 1
elif src_substr == '__suffix__' or src_substr == '__sf__':
    src_substr = ''
    add_as_suffix = 1
elif src_substr == '__space__' or src_substr == '__sp__':
    src_substr = ' '
if src_substr == '__none__' or src_substr == '__n__':
    src_substr = ''
if dst_substr == '__remove__' or dst_substr == '__rm__':
    remove_files = 1
if dst_substr == '__space__' or dst_substr == '__sp__':
    dst_substr = ' '
elif dst_substr == '__none__' or dst_substr == '__n__':
    dst_substr = ''
if dst_substr_orig == '__root__' or dst_substr_orig == '__rf__':
    dst_substr = 'root folder name'

if convert_to_lowercase:
    print('Converting to lower case')

if add_as_prefix:
    print('Adding {:s} as prefix'.format(dst_substr))
elif add_as_suffix:
    print('Adding {:s} as suffix'.format(dst_substr))
elif search_all:
    pass
elif remove_files:
    print('Searching for {:s} to remove in {:s}'.format(src_substr, src_dir))
else:
    print('Searching for {:s} to replace with {:s} in {:s}'.format(src_substr, dst_substr, src_dir))

if re_mode:
    src_substr = r'{:s}'.format(src_substr)
    print('Using regular expression mode')

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
print('Looking for matches')
iter_id = 0
for root, dirnames, filenames in os.walk(src_dir):
    if re_mode:
        if include_folders:
            for dirname in dirnames:
                matches = re.findall(src_substr, dirname)
                if matches:
                    print('{} :: {}'.format(dirname, matches))
                    src_file_paths.append(os.path.join(root, dirname))
                    src_substrs.append(matches[0])
        if include_folders != 2:
            for filename in filenames:
                matches = re.findall(src_substr, filename)
                if matches:
                    print('{} :: {}'.format(filename, matches))
                    src_file_paths.append(os.path.join(root, filename))
                    src_substrs.append(matches[0])
    else:
        print('{}'.format(root))
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

    if remove_files:
        if show_names:
            print('removing {:s}'.format(src_path))
        os.remove(src_path)
        continue

    if dst_substr_orig == '__root__' or dst_substr_orig == '__rf__':
        dst_substr = os.path.basename(os.path.dirname(src_path)) + '_'

    src_dir = os.path.dirname(src_path)
    src_fname = os.path.basename(src_path)
    src_fname_no_ext, src_ext = os.path.splitext(src_fname)
    if add_as_prefix:
        if src_fname_no_ext.startswith(dst_substr):
            print('Skipping {} that already has the required prefix {}'.format(src_fname, dst_substr))
            continue
        dst_path = os.path.join(src_dir, '{:s}{:s}{:s}'.format(dst_substr, src_fname_no_ext, src_ext))
    elif add_as_suffix:
        if src_fname_no_ext.endswith(dst_substr):
            print('Skipping {} that already has the required suffix {}'.format(src_fname, dst_substr))
            continue
        dst_path = os.path.join(src_dir, '{:s}{:s}{:s}'.format(src_fname_no_ext, dst_substr, src_ext))
    else:
        if re_mode:
            src_substr = src_substrs[src_id]

        if include_ext:
            dst_fname = src_fname.replace(src_substr, dst_substr)
            if convert_to_lowercase:
                dst_fname = dst_fname.lower()
            dst_path = os.path.join(src_dir, dst_fname)
        else:
            if re_mode:
                src_substr = src_substrs[src_id]
            dst_fname_no_ext = src_fname_no_ext.replace(src_substr, dst_substr)
            if convert_to_lowercase:
                dst_fname_no_ext = dst_fname_no_ext.lower()

            # print(f'{src_substr} --> {dst_substr}\t{src_fname_no_ext} --> {dst_fname_no_ext}')

            dst_path = os.path.join(src_dir, '{:s}{:s}'.format(dst_fname_no_ext, src_ext))
            # dst_path = src_path.replace(src_substr, dst_substr)

    src_fname_dir = os.path.dirname(src_path)
    dst_fname_dir = os.path.dirname(dst_path)
    #
    # if src_fname_dir != dst_fname_dir:
    # if show_names:
    # print 'renaming folder {:s} to {:s}'.format(src_fname_dir, dst_fname_dir)
    #         os.rename(src_fname_dir, dst_fname_dir)
    #     dst_fname = src_fname.replace(src_substr, dst_substr)

    rename_dict[src_path] = dst_path

    if write_log:
        log_fid.write('{}\t{}\n'.format(src_path, dst_path))
    if os.path.isfile(dst_path) and os.path.exists(dst_path):
        if replace_existing:
            if src_path != dst_path:
                print('Destination file: {:s} already exists. Removing it...'.format(dst_path))
                os.remove(dst_path)
        else:
            print('Destination file: {:s} already exists. Skipping it...'.format(dst_path))
            continue
    if src_path != dst_path:
        if show_names:
            print('renaming {:s} to {:s}'.format(src_path, dst_path))

        if write_log == 2:
            continue
        try:
            shutil.move(src_path, dst_path)
            # os.rename(src_path, dst_path)
        except BaseException as e:
            print('Renaming failed: {}'.format(e))
    # print matches

if write_log:
    log_fid.close()
