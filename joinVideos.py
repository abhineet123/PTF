import os
import cv2
import sys
import time
import imageio
import subprocess
import numpy as np
from pprint import pformat
from pykalman import KalmanFilter

from Misc import sortKey, processArguments, drawBox

params = {
    'db_root_dir': '',
    'actor': '',
    'seq_name': '.',
    'vid_fmt': '',
    'dst_dir': '',
    'show_img': 0,
    'n_frames': 0,
    'evenly_spaced': 0,
    'crop': 0,
    'reverse': 0,
    'roi': [],
    'resize_factor': 1.0,
    'start_id': 0,
    'out_fname_templ': 'image%06d',
    'ext': 'jpg',
    'mode': 0,
    'recursive': 1,
    'tracker_type': 0,
    'filtering': 0,
    'method': 1,
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)

    db_root_dir = params['db_root_dir']
    actor = params['actor']
    _seq_name = params['seq_name']
    show_img = params['show_img']
    vid_fmt = params['vid_fmt']
    n_frames = params['n_frames']
    evenly_spaced = params['evenly_spaced']
    roi = params['roi']
    resize_factor = params['resize_factor']
    dst_dir = params['dst_dir']
    start_id = params['start_id']
    out_fname_templ = params['out_fname_templ']
    crop = params['crop']
    reverse = params['reverse']
    ext = params['ext']
    mode = params['mode']
    recursive = params['recursive']
    tracker_type = params['tracker_type']
    filtering = params['filtering']
    method = params['method']

    vid_exts = ['.mkv', '.mp4', '.avi', '.mjpg', '.wmv', '.gif', '.webm']
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

    roi_enabled = False
    if roi and isinstance(roi, (list, tuple)) and len(roi) == 4:
        xmin, ymin, xmax, ymax = roi
        if xmax > xmin and ymax > ymin:
            print('Using roi: ', roi)
            roi_enabled = True

    # _seq_name = os.path.abspath(_seq_name)

    if os.path.isdir(_seq_name):
        if mode == 0:
            print('Looking for source videos in: {}'.format(_seq_name))
            if recursive:
                print('searching recursively')
                video_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                                   os.path.splitext(f.lower())[1] in vid_exts]
                                  for (dirpath, dirnames, filenames) in os.walk(_seq_name, followlinks=True)]
                seq_names = [item for sublist in video_file_gen for item in sublist]
            else:
                seq_names = [os.path.join(_seq_name, k) for k in os.listdir(_seq_name) for _ext in vid_exts if
                             k.endswith(_ext)]
        else:
            print('Looking for source image sequences in: {}'.format(_seq_name))
            if recursive:
                print('searching recursively')
                video_file_gen = [[os.path.join(dirpath, d) for d in dirnames if
                                   any([os.path.splitext(f.lower())[1] in img_exts
                                        for f in os.listdir(os.path.join(dirpath, d))])]
                                  for (dirpath, dirnames, filenames) in os.walk(_seq_name, followlinks=True)]
                seq_names = [item for sublist in video_file_gen for item in sublist]
            else:
                seq_names = [os.path.join(_seq_name, k) for k in os.listdir(_seq_name) if
                             any([os.path.splitext(f.lower())[1] in img_exts
                                  for f in os.listdir(os.path.join(_seq_name, k))
                                  ])]
            if not seq_names:
                seq_names = [_seq_name]

        n_videos = len(seq_names)
        if n_videos <= 0:
            raise SystemError('No input videos found')

        print('n_videos: {}'.format(n_videos))
    else:
        seq_names = [_seq_name]

    seq_names.sort()

    print('seq_names: {}'.format(seq_names))

    if reverse:
        print('Reversing videos')

    cmd = ['ffmpeg',
           '-hide_banner',
           # '-loglevel', 'panic',
           ]
    filter_str = ''

    intermediate_files = []
    for __id, seq_name in enumerate(seq_names):
        src_path = seq_name
        if vid_fmt:
            src_path = src_path + '.' + vid_fmt
        if actor:
            print('actor: ', actor)
            src_path = os.path.join(actor, src_path)
        if db_root_dir:
            print('db_root_dir: ', db_root_dir)
            src_path = os.path.join(db_root_dir, src_path)

        if mode == 0 and not os.path.isfile(src_path):
            raise IOError('Invalid video file: {}'.format(src_path))

        print('seq_name: ', seq_name)
        print('start_id: ', start_id)
        print('Reading video file: {:s}'.format(src_path))

        if method == 0:
            cmd = ['ffmpeg',
                   '-hide_banner',
                   # '-loglevel', 'panic',
                   '-i', f'{src_path}',
                   '-c', 'copy',
                   '-bsf:v', 'h264_mp4toannexb',
                   '-f', 'mpegts',
                   f'intermediate{__id}.ts']
            subprocess.run(cmd)
            intermediate_files.append(f'intermediate{__id}.ts')
        else:
            cmd.append('-i')
            cmd.append(src_path)
            filter_str += f'[{__id}:v:0][{__id}:a:0]'

    if method == 0:
        concat_str = 'concat:' + '|'.join(intermediate_files)
        cmd = ['ffmpeg',
               '-hide_banner',
               # '-loglevel', 'panic',
               '-i', concat_str,
               '-c', 'copy',
               '-bsf:a', 'aac_lctoasc',
               '-f', 'mpegts',
               f'concat_output.mp4']
    else:
        filter_str += f'concat=n={n_videos}:v=1:a=1[outv][outa]'
        cmd += [
            '-filter_complex', f'{filter_str}',
            '-map', '[outv]',
            '-map', '[outa]',
            'concat_output.mp4'
        ]


    print('cmd: {}'.format(' '.join(cmd)))

    subprocess.run(cmd)
