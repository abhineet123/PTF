import xml.etree.cElementTree as ET
import sys
import os
import glob
import numpy as np
import cv2

from pprint import pprint
from tqdm import tqdm

import paramparse

from PIL import Image

from Misc import ParamDict, drawBox, col_rgb, resizeAR, linux_path, stackImages


class Params:
    def __init__(self):
        self.cfg = ('',)
        self.end_id = -1
        self.ignore_missing_gt = 1
        self.ignore_missing_seg = 1
        self.ignored_region_only = 0
        self.mode = 0
        # self.obj_size = (50, 50)
        self.quality = 3
        self.resize = 0
        self.root_dir = '/data'
        self.show_img = 0
        self.speed = 0.5
        self.start_id = 0
        self.write_img = 1


def main():
    _params = Params()

    paramparse.process(_params)

    root_dir = _params.root_dir
    start_id = _params.start_id
    end_id = _params.end_id
    write_img = _params.write_img
    show_img = _params.show_img
    # obj_size = _params.obj_size
    ignore_missing_gt = _params.ignore_missing_gt
    ignore_missing_seg = _params.ignore_missing_seg

    params = ParamDict()

    actor = 'CTC'
    actor_sequences_dict = params.sequences_ctc

    actor_sequences = list(actor_sequences_dict.keys())

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

    ann_cols = ('green', 'blue', 'red', 'cyan', 'magenta', 'gold', 'purple', 'peach_puff', 'azure',
                'dark_slate_gray', 'navy', 'turquoise')

    out_img_root_path = linux_path(root_dir, actor, 'Images')
    os.makedirs(out_img_root_path, exist_ok=True)

    out_gt_root_path = linux_path(root_dir, actor, 'Annotations')
    os.makedirs(out_gt_root_path, exist_ok=True)

    n_frames_out_file = linux_path(root_dir, actor, 'n_frames.txt')
    n_frames_out_fid = open(n_frames_out_file, 'w')

    _exit = 0
    _pause = 1

    tif_root_dir = linux_path(root_dir, actor, 'tif')
    assert os.path.exists(tif_root_dir), "tif_root_dir does not exist"

    for seq_id in range(start_id, end_id + 1):

        seq_name = actor_sequences[seq_id]

        obj_size = actor_sequences_dict[seq_name]

        seq_img_path = linux_path(tif_root_dir, seq_name)
        assert os.path.exists(seq_img_path), "seq_img_path does not exist"

        gt_available = 1
        seq_gt_path = linux_path(tif_root_dir, seq_name + '_GT', 'TRA')
        if not os.path.exists(seq_gt_path):
            msg = "seq_gt_path does not exist"
            if ignore_missing_gt:
                print(msg)
                gt_available = 0
            else:
                raise AssertionError(msg)

        seg_available = 1
        seq_seg_path = linux_path(tif_root_dir, seq_name + '_GT', 'SEG')
        if not os.path.exists(seq_seg_path):
            msg = "seq_seg_path does not exist"
            if ignore_missing_seg:
                print(msg)
                seg_available = 0
            else:
                raise AssertionError(msg)

        seq_img_src_files = [k for k in os.listdir(seq_img_path) if
                             os.path.splitext(k.lower())[1] in img_exts]
        seq_img_src_files.sort()

        out_gt_fid = None

        if gt_available:
            seq_gt_src_files = [k for k in os.listdir(seq_gt_path) if
                                os.path.splitext(k.lower())[1] in img_exts]
            seq_gt_src_files.sort()

            assert len(seq_img_src_files) == len(
                seq_gt_src_files), "mismatch between the lengths of seq_img_src_files and seq_gt_src_files"

            out_gt_path = linux_path(out_gt_root_path, seq_name + '.txt')
            out_gt_fid = open(out_gt_path, 'w')

        if seg_available:
            seq_seq_src_files = [k for k in os.listdir(seq_seg_path) if
                                os.path.splitext(k.lower())[1] in img_exts]
            



        out_img_dir_path = linux_path(out_img_root_path, seq_name)
        os.makedirs(out_img_dir_path, exist_ok=True)

        n_frames = len(seq_img_src_files)

        print('seq {} / {}\t{}\t{} frames'.format(seq_id + start_id + 1, end_id + 1, seq_name, n_frames))

        n_frames_out_fid.write("{:d}: ('{:s}', {:d}),\n".format(seq_id, seq_name, n_frames))

        n_frames_list.append(n_frames)

        for frame_id in tqdm(range(n_frames)):
            seq_img_src_file = seq_img_src_files[frame_id]

            # assert seq_img_src_file in seq_gt_src_file, \
            #     "mismatch between seq_img_src_file and seq_gt_src_file"

            seq_img_src_file_id = int(''.join(k for k in seq_img_src_file if k.isdigit()))

            seq_img_src_path = os.path.join(seq_img_path, seq_img_src_file)

            seq_img_pil = Image.open(seq_img_src_path)

            seq_img = np.array(seq_img_pil)

            if write_img:
                out_img_file = os.path.splitext(seq_img_src_file)[0] + '.jpg'
                out_img_file_path = linux_path(out_img_dir_path, out_img_file)
                cv2.imwrite(out_img_file_path, seq_img)

            if not gt_available:
                continue

            seq_gt_src_file = seq_gt_src_files[frame_id]
            seq_gt_src_file_id = int(''.join(k for k in seq_gt_src_file if k.isdigit()))

            assert seq_gt_src_file_id == seq_img_src_file_id, \
                "Mismatch between seq_gt_src_file_id and seq_img_src_file_id"

            seq_gt_src_path = os.path.join(seq_gt_path, seq_gt_src_file)
            seq_gt_pil = Image.open(seq_gt_src_path)
            seq_gt = np.array(seq_gt_pil)
            unique_labels, counts = np.unique(seq_gt, return_counts=True)

            if show_img:
                seq_img_col = seq_img.copy()
                if len(seq_img_col.shape) == 2:
                    seq_img_col = cv2.cvtColor(seq_img_col, cv2.COLOR_GRAY2BGR)

                seq_img_col2 = seq_img_col.copy()

            size_x, size_y = obj_size, obj_size

            for label in unique_labels:
                if label == 0:
                    continue
                label_locations = np.nonzero(seq_gt == label)

                centroid_y, centroid_x = [np.mean(k) for k in label_locations]

                ymin, xmin = centroid_y - size_y / 2.0, centroid_x - size_x / 2.0
                ymax, xmax = centroid_y + size_y / 2.0, centroid_x + size_x / 2.0

                width = int(xmax - xmin)
                height = int(ymax - ymin)

                if show_img:
                    col_id = (label - 1) % len(ann_cols)

                    col = col_rgb[ann_cols[col_id]]
                    drawBox(seq_img_col, xmin, ymin, xmax, ymax, label=str(label), box_color=col)
                    seq_img_col2[label_locations] = col

                out_gt_fid.write('{:d},{:d},{:.3f},{:.3f},{:d},{:d},1,-1,-1,-1\n'.format(
                    frame_id + 1, label, xmin, ymin, width, height))

                # print()

            skip_seq = 0

            if show_img:
                seq_img_disp = stackImages([seq_img_col, seq_img_col2])

                seq_img_disp = resizeAR(seq_img_disp, height=1050, width=1900)

                cv2.imshow('seq_img_disp', seq_img_disp)
                k = cv2.waitKey(1 - _pause)
                if k == 32:
                    _pause = 1 - _pause
                elif k == 27:
                    skip_seq = 1
                    break
                elif k == ord('q'):
                    break

            if skip_seq or _exit:
                break

        if out_gt_fid is not None:
            out_gt_fid.close()

        if _exit:
            break

    n_frames_out_fid.close()


if __name__ == '__main__':
    main()
