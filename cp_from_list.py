import os, sys, glob, re
import time
from datetime import datetime

from Misc import processArguments, sortKey


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


if __name__ == '__main__':
    params = {
        'list_file': '',
        'file_name': '',
        'root_dir': '.',
        'dst_path': '',
    }
    processArguments(sys.argv[1:], params)
    list_file = params['list_file']
    root_dir = params['root_dir']
    file_name = params['file_name']
    dst_path = params['dst_path']

    if not dst_path or not os.path.isdir(dst_path):
        raise IOError('dst_path is invalid: {}'.format(dst_path))

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    if list_file:
        if os.path.isdir(list_file):
            src_paths = [os.path.join(list_file, name) for name in os.listdir(list_file) if
                         os.path.isdir(os.path.join(list_file, name))]
            src_paths.sort(key=sortKey)
        else:
            src_paths = [x.strip() for x in open(list_file).readlines() if x.strip() and not x.startswith('#')]
            if root_dir:
                root_dir = os.path.abspath(root_dir)
                src_paths = [os.path.join(root_dir, name) for name in src_paths]
        out_file_path = '{}_{}.out'.format(list_file, time_stamp)
    else:
        src_paths = [file_name]
        out_file_path = '{}_{}.out'.format(file_name, time_stamp)

    n_src_paths = len(src_paths)
    src_to_size = {}
    total_size = 0

    print("Calculating copied size...")

    for i, src_path in enumerate(src_paths):
        if src_path.startswith('#'):
            continue
        if not os.path.exists(src_path):
            raise IOError('src_path does not exist: {}'.format(src_path))
        # cp_cmd = 'rsync -r -ah --progress "{}" "{}"'.format(src_path, dst_path)

        if os.path.isdir(src_path):
            size = get_size(src_path)
        elif os.path.isfile(src_path):
            size = os.path.getsize(src_path)
        else:
            raise IOError('Invalid src_path: {}'.format(src_path))

        size_mb = size / 1e9
        total_size += size_mb

        print("{}\n{:.6f} / {:.6f} GB\n".format(src_path, size_mb, total_size))

        src_to_size[src_path] = size_mb

    done_size = 0
    speed = 0
    start_t = time.time()
    for i, src_path in enumerate(src_paths):
        if src_path.startswith('#'):
            continue
        if not os.path.exists(src_path):
            raise IOError('src_path does not exist: {}'.format(src_path))

        cp_cmd = 'cp -r "{}" "{}"'.format(src_path, dst_path)
        # cp_cmd = 'rsync -r "{}" "{}"'.format(src_path, dst_path)
        done_pc = (done_size / total_size) * 100

        print('\n{}/{} ({:.6f} GB) :: running: {}'.format(i + 1, n_src_paths, src_to_size[src_path], cp_cmd))
        os.system(cp_cmd)

        if not os.path.exists(dst_path):
            raise IOError('Copying seems to have failed as the dst_path does not exist: {}'.format(dst_path))

        done_size += src_to_size[src_path]
        end_t = time.time()
        time_taken = end_t - start_t
        speed = done_size / time_taken * 1000
        print('done {:.6f} / {:.6f} GB ({:.2f}%% in {:.2f} s @ {:.3f} MB/s)\n'.format(
            done_size, total_size, done_pc, time_taken, speed))

        src_path_full = os.path.abspath(src_path)
        with open(out_file_path, "a") as fid:
            fid.write(src_path_full + "\n")
