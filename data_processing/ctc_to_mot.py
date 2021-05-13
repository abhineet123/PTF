import xml.etree.cElementTree as ET
import sys
import os
import glob
import numpy as np
import cv2

from pprint import pprint

import paramparse

from PIL import Image

from Misc import ParamDict, drawBox, resizeAR, imshow, linux_path, stackImages


def main():
    _params = {
        'root_dir': '/data',
        'start_id': 0,
        'end_id': -1,
        'ignored_region_only': 0,
        'speed': 0.5,
        'show_img': 1,
        'quality': 3,
        'resize': 0,
        'mode': 0,
        'auto_progress': 0,
    }

    bkg_occl_seq = [24, 28, 47, 48, 49, 54]

    paramparse.process_dict(_params)

    root_dir = _params['root_dir']
    start_id = _params['start_id']
    end_id = _params['end_id']
    show_img = _params['show_img']

    params = ParamDict()

    actor = 'CTC'
    actor_sequences = params.sequences_ctc

    if end_id <= start_id:
        end_id = len(actor_sequences) - 1

    print('root_dir: {}'.format(root_dir))
    print('start_id: {}'.format(start_id))
    print('end_id: {}'.format(end_id))

    print('actor: {}'.format(actor))
    print('actor_sequences: {}'.format(actor_sequences))
    img_exts = ('.tif',)

    n_frames_list = []
    _pause = 1
    __pause = 1

    for seq_id in range(start_id, end_id + 1):
        seq_name = actor_sequences[seq_id]
        seq_img_path = linux_path(root_dir, actor, seq_name)
        seq_gt_path = linux_path(root_dir, actor, seq_name + '_GT')

        assert os.path.exists(seq_img_path), "seq_img_path does not exist"
        assert os.path.exists(seq_gt_path), "seq_gt_path does not exist"

        seq_img_src_files = [k for k in os.listdir(seq_img_path) if
                             os.path.splitext(k.lower())[1] in img_exts]
        seq_img_src_files.sort()

        seq_gt_src_files = [k for k in os.listdir(seq_gt_path) if
                            os.path.splitext(k.lower())[1] in img_exts]
        seq_gt_src_files.sort()

        assert len(seq_img_src_files) == len(
            seq_gt_src_files), "mismatch between the lengths of seq_img_src_files and seq_gt_src_files"

        n_files = len(seq_img_src_files)

        for file_id in range(n_files):
            seq_img_src_file = seq_img_src_files[file_id]
            seq_gt_src_file = seq_gt_src_files[file_id]

            # assert seq_img_src_file in seq_gt_src_file, \
            #     "mismatch between seq_img_src_file and seq_gt_src_file"

            seq_img_src_path = os.path.join(seq_img_path, seq_img_src_file)
            seq_gt_src_path = os.path.join(seq_gt_path, seq_gt_src_file)

            seq_img_pil = Image.open(seq_img_src_path)
            seq_gt_pil = Image.open(seq_gt_src_path)

            seq_img = np.array(seq_img_pil)
            seq_gt = np.array(seq_gt_pil)

            if show_img:
                seq_img_disp = stackImages([seq_img, seq_gt])

                cv2.imshow('seq_img_disp', seq_img_disp)
                cv2.waitKey(0)


if __name__ == '__main__':
    main()
