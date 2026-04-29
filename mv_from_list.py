import os, sys, glob, re
import time
import paramparse
from datetime import datetime


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


class Params:
    def __init__(self):
        self.cfg = ()
        self.dst_path = ''
        self.file_name = ''
        self.list_file = ''
        self.root_dir = '.'


if __name__ == '__main__':

    params:Params = paramparse.process(Params)

    list_file = params.list_file
    root_dir = params.root_dir
    file_name = params.file_name
    dst_path = params.dst_path

    if not dst_path or not os.path.isdir(dst_path):
        raise IOError('dst_path is invalid: {}'.format(dst_path))

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    root_dir_name = ''

    if os.path.isdir(list_file):
        src_paths = [os.path.join(list_file, name) for name in os.listdir(list_file) if
                     os.path.isdir(os.path.join(list_file, name))]
        src_paths.sort()
    else:
        src_paths = [x.strip() for x in open(list_file).readlines() if x.strip()
                     and not x.startswith('#')
                     and not x.startswith('@')]
        if root_dir:
            root_dir = os.path.abspath(root_dir)
            root_dir_name = os.path.basename(root_dir)

            src_paths = [os.path.join(root_dir, name) for name in src_paths]
    out_file_path = '{}_{}'.format(list_file, time_stamp)

    n_src_paths = len(src_paths)

    speed = 0
    start_t = time.time()
    src_path_txt = ''
    for i, src_path in enumerate(src_paths):
        if src_path.startswith('#'):
            continue
        if not os.path.exists(src_path):
            raise IOError('src_path does not exist: {}'.format(src_path))

        mv_cmd = f'mv "{src_path}" "{dst_path}/"'

        print(f'\n{i + 1}/{n_src_paths} :: running: {mv_cmd}')
        os.system(mv_cmd)
