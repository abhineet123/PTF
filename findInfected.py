import os
import sys
import shutil

import re
from pprint import pformat


def lreplace(pattern, sub, string):
    """
    Replaces 'pattern' in 'string' with 'sub' if 'pattern' starts 'string'.
    """
    return re.sub('^%s' % pattern, sub, string)


def rreplace(pattern, sub, string):
    """
    Replaces 'pattern' in 'string' with 'sub' if 'pattern' ends 'string'.
    """
    return re.sub('%s$' % pattern, sub, string)


infected_exts = ('.Adame',)
src_root_dir = '.'
dst_root_dir = 'infected'

src_root_dir = os.path.abspath(src_root_dir)
dst_root_dir = os.path.abspath(dst_root_dir)

if not src_root_dir.endswith(os.sep):
    src_root_dir += os.sep
if not dst_root_dir.endswith(os.sep):
    dst_root_dir += os.sep

if not os.path.isdir(dst_root_dir):
    os.makedirs(dst_root_dir)

print('Looking in {} for infected_files with exts {}'.format(src_root_dir, pformat(infected_exts)))
n_files = 0
n_infected_files = 0
dst_files = []
infected_files = []

for (dirpath, dirnames, filenames) in os.walk(src_root_dir, followlinks=False):
    if dst_root_dir in dirpath:
        continue

    for f in filenames:
        n_files += 1
        src_path = os.path.join(dirpath, f)

        if src_path.startswith(dst_root_dir):
            continue

        if not src_path.endswith(infected_exts):
            continue

        n_infected_files += 1
        dst_path = src_path.replace(src_root_dir, dst_root_dir)

        if src_path == dst_path:
            continue

        dst_dir = os.path.dirname(dst_path)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)

        print('{}\n\t{}\n\t{}\n'.format(n_infected_files, src_path, dst_path))

        try:
            shutil.move(src_path, dst_path)
        except PermissionError as e:
            print('\nError in moving: {}\n'.format(e))

        dst_files.append(src_path)
        dst_files.append(dst_path)

# src_file_gen = [[os.path.join(dirpath, f) for f in filenames]
#                 for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)]
# _src_files = [item for sublist in src_file_gen for item in sublist]
#
# infected_files = [f for f in _src_files if f.endswith(infected_exts)]
#
# n_files = len(_src_files)
# n_infected_files = len(infected_files)

print('Found {} infected_files among {} src_files in {}'.format(
    n_infected_files, n_files, src_root_dir))

if infected_files:
    with open('infected_files.txt', 'w') as fid:
        fid.write(str('\n'.join(infected_files).encode("utf-8")))

    # dst_files = [k.replace(src_dir, dst_dir) for k in _src_files]

    with open('infected_files_dst.txt', 'w') as fid:
        fid.write(str('\n'.join(dst_files).encode("utf-8")))
