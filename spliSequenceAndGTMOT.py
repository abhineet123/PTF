from Misc import getParamDict
from Misc import readTrackingDataMOT
from Misc import writeCornersMOT
from Misc import getFileList
import os
import sys
import shutil
import numpy as np

if __name__ == '__main__':

    params_dict = getParamDict()
    mot_actors = params_dict['mot_actors']
    mot_sequences = params_dict['mot_sequences']

    split_images = 0
    fix_frame_ids = 1
    arranged_by_frame_id = 0

    n_split_seq = 1
    # n_split_seq = 15
    # n_split_seq = 30

    actor_id = 2
    seq_type_id = 0
    seq_id = 1

    # actor = None
    # seq_name = None
    actor = 'GRAM'
    seq_name = 'isl_1_20170620-055940'
    # seq_name = '009_2011-04-24_07-00-00'
    # seq_name = 'M-30'
    # seq_name = 'M-30-HD'
    # seq_name = 'Urban1'


    split_frame_ids = None
    # split_frame_ids = [8991, 17981, 26962, 35828, 44679, 53470, 62434, 71396, 80362, 87862, 95362, 102862]
    # split_frame_ids = [1000, 1884, 2917, 3878, 4885, 5800]
    # split_frame_ids = [805, 1568]
    # split_frame_ids = [828, 1639,2487]

    # db_root_dir = '../Datasets'
    db_root_dir = 'C:/Datasets'

    # img_name_fmt = 'frame%05d.jpg'
    img_name_fmt = 'image%06d.jpg'
    img_ext = '.jpg'

    arg_id = 1
    if len(sys.argv) > arg_id:
        actor_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1

    if actor is None:
        actor = mot_actors[actor_id]
    if seq_name is None:
        sequences = mot_sequences[actor]
        if isinstance(sequences[0], list):
            sequences = sequences[seq_type_id]
        if seq_id >= len(sequences):
            print 'Invalid dataset_id: ', seq_id
            sys.exit()
        seq_name = sequences[seq_id]

    print 'actor: ', actor
    print 'seq_id:', seq_id, 'seq_name:', seq_name

    src_dir = db_root_dir + '/' + actor + '/' + seq_name
    src_fname = db_root_dir + '/' + actor + '/' + seq_name + '/' + img_name_fmt
    ground_truth_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingDataMOT(ground_truth_fname)

    if ground_truth is None:
        exit(0)

    n_gt_lines = len(ground_truth)
    print 'n_gt_lines: ', n_gt_lines

    if not arranged_by_frame_id:
        # sort by frame IDs
        print 'Sorting ground truth by frame IDs'
        ground_truth = ground_truth[ground_truth[:,0].argsort()]
        print 'Done'

    if fix_frame_ids:
        ground_truth[:, 0] += 1

    n_gt_frames = int(ground_truth[-1][0])

    if split_images:
        img_list = getFileList(src_dir, img_ext)
        n_images = len(img_list)
        if n_images != n_gt_frames:
            msg = 'No. of frames in GT: {:d} does not match the number of images in the sequence: {:d}'.format(
                n_gt_frames, n_images)
            raise SyntaxError(msg)

    print 'no_of_frames: ', n_gt_frames

    if split_frame_ids is None:
        split_size = int(n_gt_frames / n_split_seq)
        split_frame_ids = range(split_size, n_gt_frames + 1, split_size)
        split_frame_ids[-1] = n_gt_frames
    else:
        split_frame_ids.append(n_gt_frames)
        n_split_seq = len(split_frame_ids)

    print 'Splitting sequence: {:s} into {:d} parts ending at the following frames:'.format(
        seq_name, n_split_seq)
    print split_frame_ids

    start_frame_id = 1
    curr_frame_id = 1
    gt_line_id = 0
    for split_seq_id in xrange(n_split_seq):
        end_frame_id = split_frame_ids[split_seq_id]
        if end_frame_id > n_gt_frames:
            raise StandardError('Invalid split frame: {:d}'.format(end_frame_id))
        split_seq_name = '{:s}_{:d}'.format(seq_name, split_seq_id + 1)
        if split_images:
            split_seq_dir = '{:s}/{:s}/{:s}'.format(db_root_dir, actor, split_seq_name)
            if not os.path.exists(split_seq_dir):
                os.makedirs(split_seq_dir)
        split_seq_gt_fname = '{:s}/{:s}/{:s}.txt'.format(db_root_dir, actor, split_seq_name)
        split_seq_gt_file = open(split_seq_gt_fname, 'w')
        print 'Creating split sequence {:d} that goes from {:d} to {:d}'.format(split_seq_id + 1, start_frame_id,
                                                                                end_frame_id)
        if split_images:
            for frame_id in xrange(start_frame_id, end_frame_id + 1):
                split_frame_id = frame_id - start_frame_id + 1
                src_fname = '{:s}/{:s}/{:s}/image{:06d}.jpg'.format(db_root_dir, actor, seq_name, frame_id)
                dst_fname = '{:s}/{:s}/{:s}/image{:06d}.jpg'.format(db_root_dir, actor, split_seq_name, split_frame_id)
                shutil.copyfile(src_fname, dst_fname)
        while gt_line_id < n_gt_lines:
            curr_frame_id = int(ground_truth[gt_line_id][0])
            if curr_frame_id > end_frame_id:
                break
            writeCornersMOT(split_seq_gt_file, ground_truth[gt_line_id], curr_frame_id - start_frame_id + 1)
            gt_line_id += 1
        start_frame_id = end_frame_id + 1
        # for frame_id in xrange(start_frame_id, end_frame_id + 1):
        #     curr_gt = np.copy(ground_truth[ground_truth[:, 0] == frame_id])
        #     curr_gt[:, 0] -= (start_frame_id - 1)
        #     for gt in curr_gt:
        #         writeCornersMOT(split_seq_gt_file, gt)
        split_seq_gt_file.close()















