import os
import numpy as np
from datetime import datetime
from pprint import pformat

import paramparse


def main():
    params = {
        'src_path': '.',
        'save_path': '',
        'save_root_dir': '',
        'img_ext': 'jpg',
        'show_img': 1,
        'del_src': 0,
        'start_id': 0,
        'n_frames': 0,
        'width': 0,
        'height': 0,
        'fps': 30,
        # 'codec': 'FFV1',
        # 'ext': 'avi',
        'codec': 'H264',
        'ext': 'mkv',
        'out_postfix': '',
        'reverse': 0,
        'move_src': 0,
        'use_skv': 0,
        'disable_suffix': 0,
        'read_in_batch': 1,
        'placement_type': 1,
        'recursive': 1,
        'timedelta_thresh': 0.4,
    }

    paramparse.process_dict(params)
    _src_path = params['src_path']
    save_path = params['save_path']
    img_ext = params['img_ext']
    show_img = params['show_img']
    del_src = params['del_src']
    start_id = params['start_id']
    n_frames = params['n_frames']
    _width = params['width']
    _height = params['height']
    fps = params['fps']
    use_skv = params['use_skv']
    codec = params['codec']
    ext = params['ext']
    out_postfix = params['out_postfix']
    reverse = params['reverse']
    save_root_dir = params['save_root_dir']
    move_src = params['move_src']
    disable_suffix = params['disable_suffix']
    timedelta_thresh = params['timedelta_thresh']
    placement_type = params['placement_type']
    recursive = params['recursive']

    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

    src_root_dir = ''

    if os.path.isdir(_src_path):
        if recursive:
            src_paths = [_src_path]
        else:
            src_files = [k for k in os.listdir(_src_path) for _ext in img_exts if k.endswith(_ext)]
            if not src_files:
                # src_paths = [os.path.join(_src_path, k) for k in os.listdir(_src_path) if
                #              os.path.isdir(os.path.join(_src_path, k))]
                src_paths_gen = [[os.path.join(dirpath, d) for d in dirnames if
                                  any([os.path.splitext(f.lower())[1] in img_exts
                                       for f in os.listdir(os.path.join(dirpath, d))])]
                                 for (dirpath, dirnames, filenames) in os.walk(_src_path, followlinks=True)]
                src_paths = [item for sublist in src_paths_gen for item in sublist]
                src_root_dir = os.path.abspath(_src_path)
            else:
                src_paths = [_src_path]
        print('Found {} image sequence(s):\n{}'.format(len(src_paths), pformat(src_paths)))
    elif os.path.isfile(_src_path):
        print('Reading source image sequences from: {}'.format(_src_path))
        src_paths = [x.strip() for x in open(_src_path).readlines() if x.strip()]
        n_seq = len(src_paths)
        if n_seq <= 0:
            raise SystemError('No input sequences found in {}'.format(_src_path))
        print('n_seq: {}'.format(n_seq))
    else:
        raise IOError('Invalid src_path: {}'.format(_src_path))

    if recursive:
        print('searching for images recursively')

    if reverse == 1:
        print('Writing the reverse sequence')
    elif reverse == 2:
        print('Appending the reverse sequence')

    print('placement_type: {}'.format(placement_type))

    exit_prog = 0

    n_src_paths = len(src_paths)

    cwd = os.getcwd()

    for src_id, src_path in enumerate(src_paths):
        seq_name = os.path.basename(src_path)

        print('\n{}/{} Reading source images from: {}'.format(src_id + 1, n_src_paths, src_path))

        src_path = os.path.abspath(src_path)

        if move_src:
            rel_src_path = os.path.relpath(src_path, os.getcwd())
            dst_path = os.path.join(cwd, 'i2v', rel_src_path)
        else:
            dst_path = ''

        if recursive:
            src_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                             os.path.splitext(f.lower())[1] in img_exts]
                            for (dirpath, dirnames, filenames) in os.walk(src_path, followlinks=True)]
            src_files = [item for sublist in src_file_gen for item in sublist]
        else:
            src_files = [k for k in os.listdir(src_path) for _ext in img_exts if k.endswith(_ext)]

        n_src_files = len(src_files)

        if n_src_files <= 0:
            raise AssertionError('No input frames found')

        src_files.sort()
        print('n_src_files: {}'.format(n_src_files))

        src_file_names = [os.path.splitext(os.path.basename(src_file))[0] for src_file in src_files]
        src_file_name_sec = [float(src_file_name) / 1000.0 for src_file_name in src_file_names]
        src_timstamps = [datetime.fromtimestamp(src_file_name) for src_file_name in src_file_name_sec]

        src_tims_delta = [src_timstamps[i + 1] - src_timstamps[i] for i in range(n_src_files - 1)]

        src_tims_delta_seconds = [k.seconds + k.microseconds * 1e-6 for k in src_tims_delta]

        src_tims_delta_seconds.insert(0, 0)

        src_tims_delta_seconds = np.asarray(src_tims_delta_seconds)

        subseq_starts = np.where(src_tims_delta_seconds > timedelta_thresh)[0]

        subseq_lengths = subseq_starts[1:] - subseq_starts[:-1]

        subseq_lengths = list(subseq_lengths)
        subseq_lengths.insert(0, subseq_starts[0])
        subseq_lengths = np.asarray(subseq_lengths)

        np.savetxt('src_tims_delta_seconds.txt', src_tims_delta_seconds, fmt='%.6f', delimiter='\n')
        np.savetxt('subseq_starts.txt', subseq_starts, fmt='%d', delimiter='\n')
        np.savetxt('subseq_lengths.txt', subseq_lengths, fmt='%d', delimiter='\n')


if __name__ == '__main__':
    main()
