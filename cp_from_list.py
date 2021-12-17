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
        'accumulative_log': '',
        'dst_disk_label': '',
        'backup_path': '',
    }
    processArguments(sys.argv[1:], params)
    list_file = params['list_file']
    root_dir = params['root_dir']
    file_name = params['file_name']
    dst_path = params['dst_path']
    dst_disk_label = params['dst_disk_label']
    accumulative_log = params['accumulative_log']
    backup_path = params['backup_path']

    if not dst_path or not os.path.isdir(dst_path):
        raise IOError('dst_path is invalid: {}'.format(dst_path))

    abs_dst_path = os.path.abspath(dst_path).replace(os.sep, '/')
    print('abs_dst_path: {}'.format(abs_dst_path))
    dst_drive = os.path.splitdrive(abs_dst_path)[0]
    print('dst_drive: {}'.format(dst_drive))

    if not dst_disk_label:
        try:
            import win32api
        except ImportError as e:
            print('ImportError: {}'.format(e))
            pass
        else:
            dst_disk_label = win32api.GetVolumeInformation(dst_drive)
            dst_disk_label = dst_disk_label[0]

    print('dst_disk_label: {}'.format(dst_disk_label))

    # exit()

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    root_dir_name = ''

    if list_file:
        if os.path.isdir(list_file):
            src_paths = [os.path.join(list_file, name) for name in os.listdir(list_file) if
                         os.path.isdir(os.path.join(list_file, name))]
            src_paths.sort(key=sortKey)
        else:
            src_paths = [x.strip() for x in open(list_file, encoding="utf8").readlines() if x.strip()
                         and not x.startswith('#')
                         and not x.startswith('@')]
            if root_dir:
                root_dir = os.path.abspath(root_dir)
                root_dir_name = os.path.basename(root_dir)

                src_paths = [os.path.join(root_dir, name) for name in src_paths]
        out_file_path = '{}_{}'.format(list_file, time_stamp)
    else:
        src_paths = [file_name]
        out_file_path = '{}_{}'.format(file_name, time_stamp)

    if root_dir_name:
        out_file_path = '{}_{}'.format(out_file_path, root_dir_name)

    if dst_disk_label:
        out_file_path = '{}_{}'.format(out_file_path, dst_disk_label)

    out_file_path = '{}.out'.format(out_file_path)

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
    src_path_txt = ''
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
        src_path_txt += src_path_full + '\n'
        with open(out_file_path, "a") as fid:
            fid.write(src_path_full + "\n")

    if accumulative_log:
        print('writing to accumulative_log: {}'.format(accumulative_log))
        with open(accumulative_log, "a") as fid:
            fid.write('\n# {}\n'.format(time_stamp))
            if dst_disk_label and root_dir_name:
                fid.write('## {}-->{}\n'.format(root_dir_name, dst_disk_label))
            fid.write(src_path_txt + "\n")

            if backup_path:
                print('backing up accumulative_log to {}'.format(backup_path))
                cp_cmd = 'cp {} {}'.format(accumulative_log, backup_path)
                os.system(cp_cmd)


