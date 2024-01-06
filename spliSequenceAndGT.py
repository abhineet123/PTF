from Misc import getParamDict
from Misc import readDistGridParams
from Misc import readTrackingData
from Misc import writeCorners
import os
import sys
import shutil
import numpy as np

if __name__ == '__main__':

    params_dict = getParamDict()
    param_ids = readDistGridParams()
    actors = params_dict['actors']
    sequences = params_dict['sequences']

    actor_id = 2
    seq_id = 2

    # actor = None
    # seq_name = None
    actor = 'GRAM'
    seq_name = 'M-30'

    n_splits = 15

    split_frames = None
    # split_frames = [1000, 1884, 2917, 3878, 4885, 5800]
    # split_frames = [805, 1568]
    # split_frames = [828, 1639,2487]

    # db_root_dir = '../Datasets'
    db_root_dir = 'C:/Datasets'
    # img_name_fmt = 'frame%05d.jpg'
    img_name_fmt = 'image%06d.jpg'

    arg_id = 1
    if len(sys.argv) > arg_id:
        actor_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1

    if actor is None:
        actor = actors[actor_id]
    if seq_name is None:
        sequences = sequences[actor]
        if seq_id >= len(sequences):
            print 'Invalid dataset_id: ', seq_id
            sys.exit()
        seq_name = sequences[seq_id]

    src_fname = db_root_dir + '/' + actor + '/' + seq_name + '/' + img_name_fmt
    ground_truth_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)

    print 'actor: ', actor
    print 'seq_id:', seq_id, 'seq_name:', seq_name

    no_of_frames = ground_truth.shape[0]
    print 'no_of_frames: ', no_of_frames

    split_frames.append(no_of_frames)
    n_split_seq = len(split_frames)

    print 'Splitting sequence: {:s} into {:d} parts ending at the following frames:'.format(
        seq_name, n_split_seq)
    print split_frames

    start_frame_id = 0
    for split_seq_id in xrange(n_split_seq):
        end_frame = split_frames[split_seq_id]
        if end_frame>no_of_frames:
            raise StandardError('Invalid split frame: {:d}'.format(end_frame))
        split_seq_name = '{:s}_{:d}'.format(seq_name, split_seq_id + 1)
        split_seq_dir = '{:s}/{:s}/{:s}'.format(db_root_dir, actor, split_seq_name)
        if not os.path.exists(split_seq_dir):
            os.makedirs(split_seq_dir)
        split_seq_gt_fname = '{:s}/{:s}/{:s}.txt'.format(db_root_dir, actor, split_seq_name)
        split_seq_gt_file = open(split_seq_gt_fname, 'w')
        print 'Creating split sequence {:d} that goes from {:d} to {:d}'.format(split_seq_id + 1, start_frame_id + 1,
                                                                                end_frame)
        split_seq_gt_file.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
        for frame_id in xrange(start_frame_id, end_frame):
            split_frame_id = frame_id - start_frame_id
            src_fname = '{:s}/{:s}/{:s}/frame{:05d}.jpg'.format(db_root_dir, actor, seq_name, frame_id + 1)
            dst_fname = '{:s}/{:s}/{:s}/frame{:05d}.jpg'.format(db_root_dir, actor, split_seq_name, split_frame_id + 1)
            shutil.copyfile(src_fname, dst_fname)
            curr_corners = np.asarray([ground_truth[frame_id, 0:2].tolist(),
                                       ground_truth[frame_id, 2:4].tolist(),
                                       ground_truth[frame_id, 4:6].tolist(),
                                       ground_truth[frame_id, 6:8].tolist()]).T
            writeCorners(split_seq_gt_file, curr_corners, split_frame_id + 1, 0)
        start_frame_id = end_frame
        split_seq_gt_file.close()














