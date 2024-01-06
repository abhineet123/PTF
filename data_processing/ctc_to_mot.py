import xml.etree.cElementTree as ET
import sys

sys.path.append('..')

import os
import shutil
import glob
import numpy as np
import cv2

from pprint import pprint
from tqdm import tqdm
from datetime import datetime

import paramparse

from PIL import Image

from Misc import ParamDict, drawBox, col_rgb, resizeAR, linux_path, stackImages


class Params:
    def __init__(self):
        self.cfg = ('',)
        self.ignore_missing_gt = 1
        self.ignore_missing_seg = 1
        self.ignored_region_only = 0

        self.resize = 0
        self.root_dir = '/data'

        self.start_id = 34
        self.end_id = -1
        self.seq_ids = []
        # self.seq_ids = [6, 7, 14, 15]

        self.write_gt = 0
        self.write_img = 1
        self.raad_gt = 0
        self.tra_only = 0

        self.show_img = 0
        self.save_img = 0
        self.save_vid = 0

        self.disable_tqdm = 0
        self.codec = 'H264'

        self.vis_height = 1080
        self.vis_width = 1920


def main():
    _params = Params()

    paramparse.process(_params)

    root_dir = _params.root_dir
    start_id = _params.start_id
    end_id = _params.end_id

    write_img = _params.write_img
    write_gt = _params.write_gt

    save_img = _params.save_img
    save_vid = _params.save_vid
    codec = _params.codec

    show_img = _params.show_img
    vis_height = _params.vis_height
    vis_width = _params.vis_width

    # default_obj_size = _params.default_obj_size
    ignore_missing_gt = _params.ignore_missing_gt
    ignore_missing_seg = _params.ignore_missing_seg
    raad_gt = _params.raad_gt

    if save_img:
        if not show_img:
            show_img += 2
    else:
        if show_img > 1:
            save_img = 1

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

    out_img_tif_root_path = linux_path(root_dir, actor, 'Images_TIF')
    os.makedirs(out_img_tif_root_path, exist_ok=True)

    out_img_jpg_root_path = linux_path(root_dir, actor, 'Images')
    os.makedirs(out_img_jpg_root_path, exist_ok=True)

    if save_img:
        out_vis_root_path = linux_path(root_dir, actor, 'Visualizations')
        os.makedirs(out_vis_root_path, exist_ok=True)

    out_gt_root_path = linux_path(root_dir, actor, 'Annotations')
    os.makedirs(out_gt_root_path, exist_ok=True)

    n_frames_out_file = linux_path(root_dir, actor, 'n_frames.txt')
    n_frames_out_fid = open(n_frames_out_file, 'w')

    _exit = 0
    _pause = 1
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    log_path = linux_path(root_dir, actor, 'log_{}.log'.format(time_stamp))
    tif_root_dir = linux_path(root_dir, actor, 'tif')
    assert os.path.exists(tif_root_dir), "tif_root_dir does not exist"

    seq_ids = _params.seq_ids
    if not seq_ids:
        seq_ids = list(range(start_id, end_id + 1))

    n_seq = len(seq_ids)
    for __id, seq_id in enumerate(seq_ids):

        seq_name = actor_sequences[seq_id]

        default_obj_size = actor_sequences_dict[seq_name]

        seq_img_path = linux_path(tif_root_dir, seq_name)
        assert os.path.exists(seq_img_path), "seq_img_path does not exist"

        seq_img_src_files = [k for k in os.listdir(seq_img_path) if
                             os.path.splitext(k.lower())[1] in img_exts]
        seq_img_src_files.sort()

        out_gt_fid = None

        n_frames = len(seq_img_src_files)

        print('seq {} / {}\t{}\t{}\t{} frames'.format(__id+1, n_seq, seq_id, seq_name, n_frames))

        n_frames_out_fid.write("{:d}: ('{:s}', {:d}),\n".format(seq_id, seq_name, n_frames))

        n_frames_list.append(n_frames)

        gt_available = 0
        if not raad_gt:
            print('skipping GT reading')
        else:
            seq_gt_path = linux_path(tif_root_dir, seq_name + '_GT', 'TRA')
            if not os.path.exists(seq_gt_path):
                msg = "seq_gt_path does not exist"
                if ignore_missing_gt:
                    print(msg)
                else:
                    raise AssertionError(msg)
            else:
                gt_available = 1
                seq_gt_tra_file = linux_path(seq_gt_path, "man_track.txt")
                if os.path.exists(seq_gt_tra_file):
                    out_tra_file = linux_path(out_gt_root_path, seq_name + '.tra')
                    print('{} --> {}'.format(seq_gt_tra_file, out_tra_file))
                    if not os.path.exists(out_tra_file):
                        shutil.copy(seq_gt_tra_file, out_tra_file)
                    else:
                        print('skipping existing {}'.format(out_tra_file))
                else:
                    msg = "\nseq_gt_tra_file does not exist: {}".format(seq_gt_tra_file)
                    if ignore_missing_gt:
                        print(msg)
                    else:
                        raise AssertionError(msg)

        if _params.tra_only:
            continue

        seg_available = 0
        if not raad_gt:
            print('skipping segmentation reading')
        else:
            seq_seg_path = linux_path(tif_root_dir, seq_name + '_ST', 'SEG')
            if not os.path.exists(seq_seg_path):
                print("ST seq_seg_path does not exist")
                seq_seg_path = linux_path(tif_root_dir, seq_name + '_GT', 'SEG')
                if not os.path.exists(seq_seg_path):
                    msg = "GT seq_seg_path does not exist"
                    if ignore_missing_seg:
                        print(msg)
                    else:
                        raise AssertionError(msg)
                else:
                    seg_available = 1
            else:
                seg_available = 1

        if write_img:
            out_img_tif_dir_path = linux_path(out_img_tif_root_path, seq_name)
            os.makedirs(out_img_tif_dir_path, exist_ok=True)
            print('copying TIF images to {}'.format(out_img_tif_dir_path))

            out_img_jpg_dir_path = linux_path(out_img_jpg_root_path, seq_name)
            os.makedirs(out_img_jpg_dir_path, exist_ok=True)
            print('Saving jPG images to {}'.format(out_img_jpg_dir_path))

        vid_out = None
        if save_img:
            if save_vid:
                out_vis_path = linux_path(out_vis_root_path, seq_name + '.mkv')
                vid_out = cv2.VideoWriter(out_vis_path, cv2.VideoWriter_fourcc(*codec), 30, (vis_width, vis_height))
            else:
                out_vis_path = linux_path(out_vis_root_path, seq_name)
                os.makedirs(out_vis_path, exist_ok=True)
            print('Saving visualizations to {}'.format(out_vis_path))

        from collections import OrderedDict

        file_id_to_gt = OrderedDict()
        obj_id_to_gt_file_ids = OrderedDict()

        if gt_available:
            print('reading GT from {}...'.format(seq_gt_path))

            seq_gt_src_files = [k for k in os.listdir(seq_gt_path) if
                                os.path.splitext(k.lower())[1] in img_exts]
            seq_gt_src_files.sort()

            assert len(seq_img_src_files) == len(
                seq_gt_src_files), "mismatch between the lengths of seq_img_src_files and seq_gt_src_files"

            for seq_gt_src_file in tqdm(seq_gt_src_files, disable=_params.disable_tqdm):
                seq_gt_src_file_id = ''.join(k for k in seq_gt_src_file if k.isdigit())

                file_id_to_gt[seq_gt_src_file_id] = OrderedDict()

                seq_gt_src_path = os.path.join(seq_gt_path, seq_gt_src_file)
                seq_gt_pil = Image.open(seq_gt_src_path)
                seq_gt_np = np.array(seq_gt_pil)

                gt_obj_ids = list(np.unique(seq_gt_np, return_counts=False))
                gt_obj_ids.remove(0)

                for obj_id in gt_obj_ids:
                    obj_locations = np.nonzero(seq_gt_np == obj_id)
                    centroid_y, centroid_x = [np.mean(k) for k in obj_locations]

                    file_id_to_gt[seq_gt_src_file_id][obj_id] = [obj_locations, centroid_y, centroid_x]

                    if obj_id not in obj_id_to_gt_file_ids:
                        obj_id_to_gt_file_ids[obj_id] = []

                    obj_id_to_gt_file_ids[obj_id].append(seq_gt_src_file_id)

            if write_gt:
                out_gt_path = linux_path(out_gt_root_path, seq_name + '.txt')
                out_gt_fid = open(out_gt_path, 'w')

        file_id_to_seg = OrderedDict()
        file_id_to_nearest_seg = OrderedDict()

        obj_id_to_seg_file_ids = OrderedDict()
        obj_id_to_seg_sizes = OrderedDict()
        obj_id_to_seg_bboxes = OrderedDict()
        obj_id_to_mean_seg_sizes = OrderedDict()
        obj_id_to_max_seg_sizes = OrderedDict()
        all_seg_sizes = []
        mean_seg_sizes = None
        max_seg_sizes = None

        if seg_available:
            print('reading segmentation from {}...'.format(seq_seg_path))
            seq_seq_src_files = [k for k in os.listdir(seq_seg_path) if
                                 os.path.splitext(k.lower())[1] in img_exts]
            for seq_seq_src_file in tqdm(seq_seq_src_files, disable=_params.disable_tqdm):

                seq_seq_src_file_id = ''.join(k for k in seq_seq_src_file if k.isdigit())
                file_gt = file_id_to_gt[seq_seq_src_file_id]

                file_id_to_seg[seq_seq_src_file_id] = OrderedDict()

                seq_seq_src_path = os.path.join(seq_seg_path, seq_seq_src_file)
                seq_seg_pil = Image.open(seq_seq_src_path)
                seq_seg_np = np.array(seq_seg_pil)

                seg_obj_ids = list(np.unique(seq_seg_np, return_counts=False))
                seg_obj_ids.remove(0)

                _gt_obj_ids = list(file_gt.keys())

                if len(_gt_obj_ids) != len(seg_obj_ids):
                    print("\nmismatch between the number of objects in segmentation: {} and GT: {} in {}".format(
                        len(seg_obj_ids), len(_gt_obj_ids), seq_seq_src_file
                    ))

                from scipy.spatial import distance_matrix

                seg_centroids = []
                seg_id_to_locations = {}

                for seg_obj_id in seg_obj_ids:
                    # obj_id = gt_obj_ids[seg_obj_id - 1]

                    seg_obj_locations = np.nonzero(seq_seg_np == seg_obj_id)
                    seg_centroid_y, seg_centroid_x = [np.mean(k) for k in seg_obj_locations]

                    seg_centroids.append([seg_centroid_y, seg_centroid_x])

                    seg_id_to_locations[seg_obj_id] = seg_obj_locations

                gt_centroids = [[k[1], k[2]] for k in file_gt.values()]
                gt_centroids = np.asarray(gt_centroids)
                seg_centroids = np.asarray(seg_centroids)

                gt_to_seg_dists = distance_matrix(seg_centroids, gt_centroids)

                # seg_min_dist_ids = np.argmin(gt_to_seg_dists, axis=1)
                # gt_min_dist_ids = np.argmin(gt_to_seg_dists, axis=0)
                # unique_min_dist_ids = np.unique(seg_min_dist_ids)                #
                #
                # assert len(unique_min_dist_ids) == len(seg_min_dist_ids), \
                #     "duplicate matches found between segmentation and GT objects"

                # seg_to_gt_obj_ids = {
                #     seg_obj_id:  _gt_obj_ids[seg_min_dist_ids[_id]] for _id, seg_obj_id in enumerate(seg_obj_ids)
                # }

                from scipy.optimize import linear_sum_assignment

                seg_inds, gt_inds = linear_sum_assignment(gt_to_seg_dists)

                if len(seg_inds) != len(seg_obj_ids):
                    print("only {} / {} segmentation objects assigned to GT objects".format(len(seg_inds),
                                                                                            len(seg_obj_ids)))

                seg_to_gt_obj_ids = {
                    seg_obj_ids[seg_inds[i]]: _gt_obj_ids[gt_inds[i]] for i in range(len(seg_inds))
                }

                # print()

                for seg_obj_id in seg_obj_ids:
                    seg_obj_locations = seg_id_to_locations[seg_obj_id]
                    _gt_obj_id = seg_to_gt_obj_ids[seg_obj_id]

                    min_y, min_x = [np.amin(k) for k in seg_obj_locations]
                    max_y, max_x = [np.amax(k) for k in seg_obj_locations]

                    size_x, size_y = max_x - min_x, max_y - min_y

                    if _gt_obj_id not in obj_id_to_seg_sizes:
                        obj_id_to_seg_bboxes[_gt_obj_id] = []
                        obj_id_to_seg_sizes[_gt_obj_id] = []
                        obj_id_to_seg_file_ids[_gt_obj_id] = []

                    obj_id_to_seg_file_ids[_gt_obj_id].append(seq_seq_src_file_id)

                    file_id_to_seg[seq_seq_src_file_id][_gt_obj_id] = (seg_obj_locations, [min_x, min_y, max_x, max_y])

                    obj_id_to_seg_bboxes[_gt_obj_id].append([seq_seq_src_file, min_x, min_y, max_x, max_y])
                    obj_id_to_seg_sizes[_gt_obj_id].append([size_x, size_y])

                    all_seg_sizes.append([size_x, size_y])

            obj_id_to_mean_seg_sizes = OrderedDict({k: np.mean(v, axis=0) for k, v in obj_id_to_seg_sizes.items()})
            obj_id_to_max_seg_sizes = OrderedDict({k: np.amax(v, axis=0) for k, v in obj_id_to_seg_sizes.items()})

            print('segmentations found for {} files'.format
                  (len(file_id_to_seg),
                   # '\n'.join(file_id_to_seg.keys())
                   )
                  )
            print('segmentations include {} objects:\n{}'.format(
                len(obj_id_to_seg_bboxes), ', '.join(str(k) for k in obj_id_to_seg_bboxes.keys())))

            mean_seg_sizes = np.mean(all_seg_sizes, axis=0)
            max_seg_sizes = np.amax(all_seg_sizes, axis=0)

            for obj_id in obj_id_to_seg_file_ids:
                seg_file_ids = obj_id_to_seg_file_ids[obj_id]
                seg_file_ids_num = np.asarray(list(int(k) for k in seg_file_ids))
                gt_file_ids = obj_id_to_gt_file_ids[obj_id]
                # gt_file_ids_num = np.asarray(list(int(k) for k in gt_file_ids))
                gt_seg_file_ids_dist = {
                    gt_file_id: np.abs(int(gt_file_id) - seg_file_ids_num)
                    for gt_file_id in gt_file_ids
                }
                file_id_to_nearest_seg[obj_id] = {
                    gt_file_id: seg_file_ids[np.argmin(_dist).item()]
                    for gt_file_id, _dist in gt_seg_file_ids_dist.items()
                }

        nearest_seg_size = OrderedDict()

        for frame_id in tqdm(range(n_frames), disable=_params.disable_tqdm):
            seq_img_src_file = seq_img_src_files[frame_id]

            # assert seq_img_src_file in seq_gt_src_file, \
            #     "mismatch between seq_img_src_file and seq_gt_src_file"

            seq_img_src_file_id = ''.join(k for k in seq_img_src_file if k.isdigit())

            seq_img_src_path = os.path.join(seq_img_path, seq_img_src_file)

            # seq_img_pil = Image.open(seq_img_src_path)
            # seq_img = np.array(seq_img_pil)

            seq_img = cv2.imread(seq_img_src_path, cv2.IMREAD_UNCHANGED)
            # assert (seq_img == seq_img_cv).all(), "mismatch between PIL and cv2 arrays"
            # seq_img_cv_unique = np.unique(seq_img_cv)
            # n_seq_img_unique_cv = len(seq_img_cv_unique)

            # seq_img_float = seq_img.astype(np.float32) / 65535.

            max_pix, min_pix = np.amax(seq_img), np.amin(seq_img)

            seq_img_float_norm = (seq_img.astype(np.float32) - min_pix) / (max_pix - min_pix)
            seq_img_uint8 = (seq_img_float_norm * 255.).astype(np.uint8)
            # seq_img_uint8 = (seq_img / 256.).astype(np.uint8)
            max_pix_uint8, min_pix_uint8 = np.amax(seq_img_uint8), np.amin(seq_img_uint8)

            seq_img_unique = np.unique(seq_img)
            seq_img_unique_uint8 = np.unique(seq_img_uint8)

            n_seq_img_unique = len(seq_img_unique)
            n_seq_img_unique_uint8 = len(seq_img_unique_uint8)

            if n_seq_img_unique > n_seq_img_unique_uint8:
                # print('{} :: drop in number of unique values from {} ({}, {}) to {} ({}, {})'.format(
                #     seq_img_src_file, n_seq_img_unique, max_pix, min_pix,
                #     n_seq_img_unique_uint8, max_pix_uint8, min_pix_uint8))
                with open(log_path, 'a') as log_fid:
                    log_fid.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        seq_name, seq_img_src_file,
                        n_seq_img_unique, n_seq_img_unique_uint8,
                        max_pix, min_pix,
                        max_pix_uint8, min_pix_uint8
                    ))
                # print()

            if write_img:
                # out_img_file = os.path.splitext(seq_img_src_file)[0] + '.png'
                # out_img_file_path = linux_path(out_img_tif_dir_path, out_img_file)

                out_img_file_tif = os.path.splitext(seq_img_src_file)[0] + '.tif'
                out_img_file_path_tif = linux_path(out_img_tif_dir_path, out_img_file_tif)

                out_img_file_uint8 = os.path.splitext(seq_img_src_file)[0] + '.jpg'
                out_img_file_path_uint8 = linux_path(out_img_jpg_dir_path, out_img_file_uint8)

                if not os.path.exists(out_img_file_path_uint8):
                    # cv2.imwrite(out_img_file_path, seq_img)
                    cv2.imwrite(out_img_file_path_uint8, seq_img_uint8)

                if not os.path.exists(out_img_file_path_tif):
                    # print('{} --> {}'.format(seq_img_src_path, out_img_file_path_tif))
                    shutil.copyfile(seq_img_src_path, out_img_file_path_tif)

            if show_img:
                seq_img_col = seq_img_uint8.copy()
                if len(seq_img_col.shape) == 2:
                    seq_img_col = cv2.cvtColor(seq_img_col, cv2.COLOR_GRAY2BGR)

                # seq_img_col2 = seq_img_col.copy()

                seq_img_col3 = seq_img_col.copy()

            if not gt_available:
                if save_img:
                    if vid_out is not None:
                        vid_out.write(seq_img_col)
                continue

            file_gt = file_id_to_gt[seq_img_src_file_id]

            # seq_gt_src_file = seq_gt_src_files[frame_id]
            # assert seq_gt_src_file_id == seq_img_src_file_id, \
            #     "Mismatch between seq_gt_src_file_id and seq_img_src_file_id"

            gt_obj_ids = list(file_gt.keys())
            for obj_id in gt_obj_ids:
                assert obj_id != 0, "invalid object ID"

                try:
                    nearest_seg_file_ids = file_id_to_nearest_seg[obj_id]
                except KeyError:
                    file_seg = {}
                else:
                    nearest_seg_file_id = nearest_seg_file_ids[seq_img_src_file_id]
                    file_seg = file_id_to_seg[nearest_seg_file_id]

                obj_locations, centroid_y, centroid_x = file_gt[obj_id]

                if file_seg:
                    xmin, ymin, xmax, ymax = file_seg[obj_id][1]
                    size_x, size_y = xmax - xmin, ymax - ymin
                    nearest_seg_size[obj_id] = (size_x, size_y)
                else:
                    try:
                        size_x, size_y = nearest_seg_size[obj_id]
                    except KeyError:
                        try:
                            size_x, size_y = obj_id_to_mean_seg_sizes[obj_id]
                        except KeyError:
                            if mean_seg_sizes is not None:
                                size_x, size_y = mean_seg_sizes
                            else:
                                size_x, size_y = default_obj_size, default_obj_size

                    ymin, xmin = centroid_y - size_y / 2.0, centroid_x - size_x / 2.0
                    ymax, xmax = centroid_y + size_y / 2.0, centroid_x + size_x / 2.0

                width = int(xmax - xmin)
                height = int(ymax - ymin)

                if show_img:
                    col_id = (obj_id - 1) % len(ann_cols)

                    col = col_rgb[ann_cols[col_id]]
                    drawBox(seq_img_col, xmin, ymin, xmax, ymax, label=str(obj_id), box_color=col)
                    # seq_img_col2[obj_locations] = col

                    if file_seg:
                        try:
                            locations, bbox = file_seg[obj_id][:2]
                        except KeyError:
                            print('weird stuff going on here')
                        else:
                            seq_img_col3[locations] = col
                            # min_x, min_y, max_x, max_y = bbox
                            # drawBox(seq_img_col3, min_x, min_y, max_x, max_y, label=str(obj_id), box_color=col)

                if write_gt:
                    out_gt_fid.write('{:d},{:d},{:.3f},{:.3f},{:d},{:d},1,-1,-1,-1\n'.format(
                        frame_id + 1, obj_id, xmin, ymin, width, height))
                # print()
            skip_seq = 0

            if show_img:
                # images_to_stack = [seq_img_col, seq_img_col2]
                images_to_stack = [seq_img_col, ]
                if file_seg:
                    images_to_stack.append(seq_img_col3)
                    __pause = _pause
                else:
                    __pause = _pause

                seq_img_vis = stackImages(images_to_stack, sep_size=5)

                seq_img_vis = resizeAR(seq_img_vis, height=vis_height, width=vis_width)

                cv2.putText(seq_img_vis, '{}: {}'.format(seq_name, seq_img_src_file_id), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if show_img != 2:
                    cv2.imshow('seq_img_vis', seq_img_vis)
                    k = cv2.waitKey(1 - __pause)
                    if k == 32:
                        _pause = 1 - _pause
                    elif k == 27:
                        skip_seq = 1
                        break
                    elif k == ord('q'):
                        break

                if save_img:
                    if vid_out is not None:
                        vid_out.write(seq_img_vis)
                    else:
                        out_vis_file = os.path.splitext(seq_img_src_file)[0] + '.jpg'
                        out_vis_file_path = linux_path(out_vis_path, out_vis_file)
                        cv2.imwrite(out_vis_file_path, seq_img_vis)

            if skip_seq or _exit:
                break
        if vid_out is not None:
            vid_out.release()

        if out_gt_fid is not None:
            out_gt_fid.close()

        if _exit:
            break

    n_frames_out_fid.close()


if __name__ == '__main__':
    main()
