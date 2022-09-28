import cv2
import sys
import time
import os
import shutil
from datetime import datetime
from pprint import pformat
from tqdm import tqdm

import multiprocessing
import functools

import threading
import paramparse
# print(paramparse.__file__)

from Misc import sortKey, linux_path


def write_roi_frame(src_file, src_frames, roi_out_path, xmin, ymin, xmax, ymax):
    frame = get_frame(src_file, src_frames)

    out_fname = os.path.basename(src_file)
    out_img_path = linux_path(roi_out_path, out_fname)

    frame_roi = frame[ymin:ymax, xmin:xmax, ...]

    cv2.imwrite(out_img_path, frame_roi)


def get_frame(src_file, src_frames):
    try:
        frame = src_frames[src_file]
    except KeyError:

        frame = cv2.imread(src_file)
        if frame is None:
            raise AssertionError(f'frame could not be read: {src_file}')
        src_frames[src_file] = frame
    return frame


def read_all_frames(src_files, src_frames, n_proc):
    print('reading all frames')

    if n_proc > 1:
        read_fn = functools.partial(
            get_frame,
            src_frames=src_frames,
        )

        print('running in parallel over {} processes'.format(n_proc))
        pool = multiprocessing.Pool(n_proc)

        pool.map(read_fn, src_files)
    else:
        for src_file in src_files:
            get_frame(src_file, src_frames)


def main():
    params = {
        'root_dir': '',
        'rois_path': '',
        'src_path': '.',
        'n_proc': 6,
    }
    paramparse.process_dict(params)

    root_dir = params['root_dir']
    _src_path = params['src_path']
    rois_path = params['rois_path']
    n_proc = params['n_proc']

    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

    if root_dir:
        _src_path = linux_path(root_dir, _src_path)
        rois_path = linux_path(root_dir, rois_path)

    if os.path.isdir(_src_path):
        src_files = [k for k in os.listdir(_src_path) for _ext in img_exts if k.endswith(_ext)]
        if not src_files:
            """Nine images in the source folder itself so look for folders containing images 
            within the source folder"""
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
        print('n_seq: {}'.format(n_seq))
    else:
        raise IOError('Invalid src_path: {}'.format(_src_path))

    rois = [x.strip() for x in open(rois_path).readlines() if x.strip()]

    rois = [[int(x) for x in roi.split('_')] for roi in rois]

    n_rois = len(rois)

    print(f'creating {n_rois} rois')

    n_src_paths = len(src_paths)

    for src_id, src_path in enumerate(src_paths):

        print('\n{}/{} Reading source images from: {}'.format(src_id + 1, n_src_paths, src_path))

        src_path = os.path.abspath(src_path)

        src_files = [linux_path(src_path, k)  for k in os.listdir(src_path) for _ext in img_exts if k.endswith(_ext)]
        n_src_files = len(src_files)

        if n_src_files <= 0:
            raise IOError('No input frames found')

        src_files.sort(key=sortKey)

        src_frames = {}

        # thread = threading.Thread(target=read_all_frames,
        #                           args=(src_files, src_frames, n_proc))
        # thread.start()

        roi_out_paths = {}

        for roi_id, roi in enumerate(rois):
            xmin, ymin, xmax, ymax = roi

            roi_out_dir = "roi_{:d}_{:d}_{:d}_{:d}".format(
                xmin, ymin, xmax, ymax)

            out_root_dir = os.path.dirname(src_path)
            roi_out_path = linux_path(out_root_dir, roi_out_dir)

            os.makedirs(roi_out_path, exist_ok=1)

            roi_out_paths[roi_id] = roi_out_path

        print('n_src_files: {}'.format(n_src_files))
        for src_file in tqdm(src_files):
            frame = cv2.imread(src_file)
            for roi_id, roi in enumerate(tqdm(rois)):
                xmin, ymin, xmax, ymax = roi

                out_fname = os.path.basename(src_file)
                out_img_path = linux_path(roi_out_paths[roi_id], out_fname)

                frame_roi = frame[ymin:ymax, xmin:xmax, ...]

                cv2.imwrite(out_img_path, frame_roi)

                # write_roi_fn = functools.partial(
                #     write_roi_frame,
                #     src_frames=src_frames,
                #     roi_out_path=roi_out_paths[roi_id],
                #     xmin=xmin,
                #     ymin=ymin,
                #     xmax=xmax,
                #     ymax=ymax,
                # )
                #
                # for src_file in tqdm(src_files):
                #     write_roi_fn(src_file)

                # print('running in parallel over {} processes'.format(n_proc))
                # pool = multiprocessing.Pool(n_proc)
                # pool.map(write_roi_fn, src_files)


if __name__ == '__main__':
    main()
