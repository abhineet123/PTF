from Misc import readTrackingData
from Misc import getParamDict
from Misc import readDistGridParams
from Misc import drawRegion

import sys
import cv2
import numpy as np
import os

if __name__ == '__main__':

    params_dict = getParamDict()
    # param_ids = readDistGridParams()
    pause_seq = 0
    gt_col = (0, 0, 255)
    gt_thickness = 2
    actors = params_dict['actors']
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d.jpg'

    opt_gt_ssm = '2r'
    use_opt_gt = 0

    actor_id = 1

    init_frame_id = 0
    end_frame_id = -1
    show_all_seq = 0
    start_id = 95
    end_id = -1

    actor = actors[actor_id]
    print 'actor: ', actor

    sequences = sequences[actor]
    n_seq = len(sequences)

    if show_all_seq:
        start_id = 0
        end_id = n_seq - 1
        print 'n_seq: ', n_seq
    elif end_id < start_id:
        end_id = start_id
    seq_ids = range(start_id, end_id + 1)

    for curr_seq_id in seq_ids:
        seq_name = sequences[curr_seq_id]
        # seq_name = 'nl_mugII_s1'
        print 'sequence {:d} : {:s}'.format(curr_seq_id, seq_name)

        if opt_gt_ssm == '0':
            print 'Using standard ground truth'
            use_opt_gt = 0
        else:
            use_opt_gt = 1
            print 'Using optimized ground truth with ssm: ', opt_gt_ssm

        actor_dir = db_root_dir + '/' + actor
        src_dir = db_root_dir + '/' + actor + '/' + seq_name

        orig_gt_corners_fname = '{:s}/{:s}/{:s}.txt'.format(db_root_dir, actor, seq_name)
        print 'Reading ground truth from:: ', orig_gt_corners_fname
        orig_ground_truth = readTrackingData(orig_gt_corners_fname)
        no_of_frames = orig_ground_truth.shape[0]
        file_list = [each for each in os.listdir(src_dir) if each.endswith('.jpg')]
        n_files = len(file_list)

        if n_files != no_of_frames:
            raise StandardError(
                'No. of frames in the ground truth: {:d} does not match the no. of images in the sequence: {:d}'.format(
                    no_of_frames, n_files))
        print 'no_of_frames: ', no_of_frames

        if end_frame_id < init_frame_id:
            end_frame_id = no_of_frames - 1

        for start_id in xrange(init_frame_id, end_frame_id + 1):
            gt_corners_window_name = '{:d} : {:s}: frame {:d}'.format(
                curr_seq_id, seq_name, start_id)
            if use_opt_gt:
                gt_corners_fname = '{:s}/ReinitGT/{:s}/frame{:05d}_{:s}.txt'.format(
                    actor_dir, seq_name, start_id + 1, opt_gt_ssm)
                gt_corners_window_name += ' GT: {:s}'.format(opt_gt_ssm)
            else:
                gt_corners_fname = '{:s}/ReinitGT/{:s}/frame{:05d}.txt'.format(
                    actor_dir, seq_name, start_id + 1)

            print 'Reading reinit ground truth from:: ', gt_corners_fname
            ground_truth = readTrackingData(gt_corners_fname)
            no_of_reinit_frames = ground_truth.shape[0]

            if no_of_reinit_frames != no_of_frames - start_id:
                raise StandardError(
                    'No. of frames in the reinit ground truth for frame {:d} : {:d} does not match the expected count: {:d}'.format(
                        start_id + 1, no_of_reinit_frames, no_of_frames - start_id))


            cv2.namedWindow(gt_corners_window_name)
            for i in xrange(no_of_reinit_frames):
                curr_corners = np.asarray([ground_truth[i, 0:2].tolist(),
                                           ground_truth[i, 2:4].tolist(),
                                           ground_truth[i, 4:6].tolist(),
                                           ground_truth[i, 6:8].tolist()]).T
                curr_img = cv2.imread('{:s}/frame{:05d}.jpg'.format(src_dir, i + start_id + 1))
                if curr_img is None:
                    raise StandardError('End of sequence reached unexpectedly')

                if len(curr_img.shape) == 2:
                    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)

                if np.isnan(curr_corners).any():
                    print 'curr_corners:\n', curr_corners
                    raise StandardError('Gound truth for frame {:d} contains NaN'.format(i))
                drawRegion(curr_img, curr_corners, gt_col, gt_thickness)

                fps_text = 'frame: {:4d}'.format(i + 1)
                cv2.putText(curr_img, fps_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
                cv2.imshow(gt_corners_window_name, curr_img)

                key = cv2.waitKey(1 - pause_seq)
                if key == 27:
                    break
                elif key == 32:
                    pause_seq = 1 - pause_seq
            cv2.destroyWindow(gt_corners_window_name)














