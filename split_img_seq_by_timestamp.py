import os
import numpy as np
from datetime import datetime
from pprint import pformat
from tqdm import tqdm
import random
import pandas as pd

import paramparse


class Params:
    """
    :ivar codec:
    :type codec: str

    :ivar del_src:
    :type del_src: int

    :ivar disable_suffix:
    :type disable_suffix: int

    :ivar ext:
    :type ext: str

    :ivar fps:
    :type fps: int

    :ivar height:
    :type height: int

    :ivar img_ext:
    :type img_ext: str

    :ivar max_rials:
    :type max_rials: int

    :ivar move_src:
    :type move_src: int

    :ivar n_frames:
    :type n_frames: int

    :ivar out_postfix:
    :type out_postfix: str

    :ivar placement_type:
    :type placement_type: int

    :ivar ratio_eps:
    :type ratio_eps: float

    :ivar read_in_batch:
    :type read_in_batch: int

    :ivar recursive:
    :type recursive: int

    :ivar reverse:
    :type reverse: int

    :ivar save_path:
    :type save_path: str

    :ivar save_root_dir:
    :type save_root_dir: str

    :ivar show_img:
    :type show_img: int

    :ivar src_path:
    :type src_path: str

    :ivar src_root_dir:
    :type src_root_dir: str

    :ivar start_id:
    :type start_id: int

    :ivar test_ratio:
    :type test_ratio: float

    :ivar timedelta_thresh:
    :type timedelta_thresh: float

    :ivar use_skv:
    :type use_skv: int

    :ivar width:
    :type width: int

    """

    def __init__(self):
        self.cfg = ()
        self.codec = 'H264'
        self.del_src = 0
        self.disable_suffix = 0
        self.ext = 'mkv'
        self.fps = 30
        self.height = 0
        self.img_ext = 'jpg'
        self.max_rials = 100000
        self.move_src = 0
        self.n_frames = 0
        self.out_postfix = ''
        self.placement_type = 1
        self.ratio_eps = 1e-3
        self.read_in_batch = 1
        self.recursive = 1
        self.reverse = 0
        self.save_path = ''
        self.save_root_dir = ''
        self.show_img = 1
        self.src_path = '.'
        self.src_root_dir = ''
        self.start_id = 0
        self.test_ratio = 0.3
        self.timedelta_thresh = 1.0
        self.use_skv = 0
        self.width = 0


def main():
    params = Params()

    paramparse.process(params)
    src_root_dir = params.src_root_dir
    _src_path = params.src_path
    reverse = params.reverse
    timedelta_thresh = params.timedelta_thresh
    recursive = params.recursive
    test_ratio = params.test_ratio
    max_rials = params.max_rials
    ratio_eps = params.ratio_eps

    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

    if os.path.isdir(_src_path):
        if recursive:
            src_paths = [_src_path]
        else:
            src_files = [k for k in os.listdir(_src_path) for _ext in img_exts if k.endswith(_ext)]
            if not src_files:
                src_paths_gen = [[os.path.join(dirpath, d) for d in dirnames if
                                  any([os.path.splitext(f.lower())[1] in img_exts
                                       for f in os.listdir(os.path.join(dirpath, d))])]
                                 for (dirpath, dirnames, filenames) in os.walk(_src_path, followlinks=True)]
                src_paths = [item for sublist in src_paths_gen for item in sublist]
            else:
                src_paths = [_src_path]
        print('Found {} image sequence(s):\n{}'.format(len(src_paths), pformat(src_paths)))
    elif os.path.isfile(_src_path):
        print('Reading source image sequences from: {}'.format(_src_path))
        src_paths = [x.strip() for x in open(_src_path).readlines() if x.strip()]
        n_seq = len(src_paths)
        if n_seq <= 0:
            raise SystemError('No input sequences found in {}'.format(_src_path))

        if src_root_dir:
            src_paths = [os.path.join(src_root_dir, x) for x in src_paths]

        print('n_seq: {}'.format(n_seq))
    else:
        raise IOError('Invalid src_path: {}'.format(_src_path))

    if recursive:
        print('searching for images recursively')

    if reverse == 1:
        print('Writing the reverse sequence')
    elif reverse == 2:
        print('Appending the reverse sequence')

    n_src_paths = len(src_paths)

    for src_id, src_path in enumerate(src_paths):
        seq_name = os.path.basename(src_path)

        print('\n{}/{} Reading source images from: {}'.format(src_id + 1, n_src_paths, src_path))

        src_path = os.path.abspath(src_path)

        if recursive:
            src_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                             os.path.splitext(f.lower())[1] in img_exts]
                            for (dirpath, dirnames, filenames) in os.walk(src_path, followlinks=True)]
            src_files = [item for sublist in src_file_gen for item in sublist]
        else:
            src_files = [k for k in os.listdir(src_path) for _ext in img_exts if k.endswith(_ext)]

        n_files = len(src_files)

        if n_files <= 0:
            raise AssertionError('No input frames found')

        src_files.sort()
        print('n_files: {}'.format(n_files))

        src_file_names = [os.path.splitext(os.path.basename(src_file))[0] for src_file in src_files]
        src_file_name_sec = [float(src_file_name) / 1000.0 for src_file_name in src_file_names]
        src_timstamps = [datetime.fromtimestamp(src_file_name) for src_file_name in src_file_name_sec]

        src_tims_delta = [src_timstamps[i + 1] - src_timstamps[i] for i in range(n_files - 1)]

        src_tims_delta_seconds = [k.seconds + k.microseconds * 1e-6 for k in src_tims_delta]
        src_tims_delta_seconds.insert(0, 0)

        subseq_start_ids = list(np.where(np.asarray(src_tims_delta_seconds) > timedelta_thresh)[0])
        subseq_start_ids.insert(0, 0)

        n_subseq = len(subseq_lengths)
        print('n_subseq: {}'.format(n_subseq))

        subseq_start_ids = np.asarray(subseq_start_ids)

        subseq_lengths = list(subseq_start_ids[1:] - subseq_start_ids[:-1])
        subseq_lengths.append(n_files - subseq_start_ids[-1])

        subseq_end_ids = subseq_start_ids + np.asarray(subseq_lengths) - 1

        subseq_timestamps = [(src_timstamps[subseq_start_ids[i]], src_timstamps[subseq_end_ids[i]])
                             for i in range(n_subseq)]

        subseq_ids_and_lengths = list(range(n_subseq))

        n_test_subseq = int(n_subseq * test_ratio)
        print('n_test_subseq: {}'.format(n_test_subseq))

        best_test_files_ratio = 0
        best_n_test_files = 0
        best_test_subseq_ids = best_train_subseq_ids = None

        pbar = tqdm(range(max_rials))

        print('looking for a random set of test sub sequences with closest possible ratio of files to {:.2f}'.format(
            test_ratio))

        for _ in pbar:
            random.shuffle(subseq_ids_and_lengths)
            test_subseq_lengths_with_ids = subseq_ids_and_lengths[:n_test_subseq]

            n_test_files = sum(k[1] for k in test_subseq_lengths_with_ids)

            test_files_ratio = float(n_test_files) / n_files

            if abs(test_ratio - test_files_ratio) < abs(test_ratio - best_test_files_ratio):
                best_test_files_ratio = test_files_ratio
                best_n_test_files = n_test_files
                best_test_subseq_ids = test_subseq_lengths_with_ids.copy()
                best_train_subseq_ids = subseq_ids_and_lengths[n_test_subseq:].copy()

                pbar.set_description('best_test_ratio: {:.10f} n_test_files: {:d}'.format(
                    best_test_files_ratio, best_n_test_files))

                if abs(best_test_files_ratio - test_ratio) < ratio_eps:
                    break

        print('n_test_files: {}'.format(best_n_test_files))

        subseq_lengths = np.asarray(subseq_lengths)

        subseq_info = [
            {
                'start_id': subseq_start_ids[i],
                'end_id': subseq_end_ids[i],
                'length': subseq_lengths[i],
                'start_timestamp': subseq_timestamps[i][0].strftime('%Y/%m/%d %I:%M:%S:%f %p'),
                'end_timestamp': subseq_timestamps[i][1].strftime('%I:%M:%S %p'),
            }
            for i in range(n_subseq)
        ]

        df = pd.DataFrame(subseq_info, columns=['start_id', 'end_id', 'length', 'start_timestamp', 'end_timestamp'])
        df.to_csv(os.path.join(src_path, f'{seq_name}_subseq_info.csv'), index = False, sep='\t')

    if __name__ == '__main__':
        main()
