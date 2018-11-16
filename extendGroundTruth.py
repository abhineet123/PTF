from Misc import readTrackingData
from Misc import getParamDict
from Misc import readDistGridParams
from Misc import drawRegion
from Misc import getTrackingObject2
from Misc import writeCorners
from Homography import compute_homography
import utility as util

import sys
import cv2
import numpy as np
import os

if __name__ == '__main__':

    params_dict = getParamDict()
    param_ids = readDistGridParams()
    pause_seq = 1
    gt_col = (0, 0, 255)
    gt_thickness = 2
    annotation_col = (0, 255, 0)
    actors = params_dict['actors']
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d.jpg'
    opt_gt_ssm = '0'
    use_opt_gt = 0
    init_sel_from_file = 0
    in_corners_fname = 'omni_magazine.br'
    out_suffix = 'ext'
    hom_from_prev = 1

    actor_id = 3
    seq_id = 26
    init_frame_id = 0

    actor = actors[actor_id]
    print 'actor: ', actor

    sequences = sequences[actor]
    n_seq = len(sequences)

    seq_name = sequences[seq_id]
    # seq_name = 'nl_mugII_s1'
    print 'sequence {:d} : {:s}'.format(seq_id, seq_name)

    if opt_gt_ssm == '0':
        print 'Using standard ground truth'
        use_opt_gt = 0
    else:
        use_opt_gt = 1
        print 'Using optimized ground truth with ssm: ', opt_gt_ssm

    src_dir = db_root_dir + '/' + actor + '/' + seq_name
    src_fname = db_root_dir + '/' + actor + '/' + seq_name + '/' + img_name_fmt

    fname = '{:s}/frame{:05d}.jpg'.format(src_dir, init_frame_id + 1)
    print 'fname: ', fname
    init_img = cv2.imread(fname)
    if len(init_img.shape) == 2:
        init_img = cv2.cvtColor(init_img, cv2.COLOR_GRAY2RGB)

    if use_opt_gt:
        gt_corners_fname = '{:s}/{:s}/OptGT/{:s}_{:s}.txt'.format(db_root_dir, actor, seq_name, opt_gt_ssm)
        sel_corners_fname = '{:s}/{:s}/OptGT/{:s}_{:s}_{:s}.txt'.format(
            db_root_dir, actor, seq_name, opt_gt_ssm, out_suffix)
    else:
        gt_corners_fname = '{:s}/{:s}/{:s}.txt'.format(db_root_dir, actor, seq_name)
        sel_corners_fname = '{:s}/{:s}/{:s}_{:s}.txt'.format(db_root_dir, actor, seq_name, out_suffix)
    print 'Reading ground truth from:: ', gt_corners_fname
    print 'Writing extended ground truth to:: ', sel_corners_fname

    ground_truth = readTrackingData(gt_corners_fname)
    if ground_truth is None:
        raise StandardError('Ground truth could not be read')
    no_of_frames = ground_truth.shape[0]

    file_list = [each for each in os.listdir(src_dir) if each.endswith('.jpg')]
    n_files = len(file_list)

    if n_files != no_of_frames:
        print 'No. of frames in the ground truth: {:d} does not match the no. of images in the sequence: {:d}'.format(
            no_of_frames, n_files)

    print 'no_of_frames: ', no_of_frames

    gt_corners_window_name = '{:d} : {:s}'.format(seq_id, seq_name)

    init_corners_gt = np.asarray([ground_truth[init_frame_id, 0:2].tolist(),
                                  ground_truth[init_frame_id, 2:4].tolist(),
                                  ground_truth[init_frame_id, 4:6].tolist(),
                                  ground_truth[init_frame_id, 6:8].tolist()]).T

    if len(init_img.shape) == 2:
        init_img = cv2.cvtColor(init_img, cv2.COLOR_GRAY2RGB)

    if init_sel_from_file:
        in_corners_file_path = '{:s}/{:s}/{:s}'.format(db_root_dir, actor, in_corners_fname)
        print 'Reading selected corners from : ', in_corners_file_path
        sel_corners = readTrackingData(in_corners_file_path)
        if sel_corners is None:
            raise StandardError('Selected Corners could not be read')
        init_corners_selected = np.asarray([sel_corners[init_frame_id, 0:2].tolist(),
                                            sel_corners[init_frame_id, 2:4].tolist(),
                                            sel_corners[init_frame_id, 4:6].tolist(),
                                            sel_corners[init_frame_id, 6:8].tolist()]).T
    else:
        init_corners_selected = getTrackingObject2(init_img,
                                                   title='Select the initial object location in the new ground truth',
                                                   line_thickness=2)
        init_corners_selected = np.array(init_corners_selected, dtype=np.float64).T

    sel_corners_fid = open(sel_corners_fname, 'w')
    sel_corners_fid.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
    writeCorners(sel_corners_fid, init_corners_selected, init_frame_id + 1)

    cv2.namedWindow(gt_corners_window_name)
    prev_corners_gt = init_corners_gt.copy()
    prev_corners_selected = init_corners_selected.copy()
    for frame_id in xrange(init_frame_id + 1, no_of_frames):
        curr_corners_gt = np.asarray([ground_truth[frame_id, 0:2].tolist(),
                                      ground_truth[frame_id, 2:4].tolist(),
                                      ground_truth[frame_id, 4:6].tolist(),
                                      ground_truth[frame_id, 6:8].tolist()]).T

        curr_hom_gt = util.compute_homography(prev_corners_gt, curr_corners_gt)
        curr_corners_selected = util.dehomogenize(np.mat(curr_hom_gt) * np.mat(util.homogenize(prev_corners_selected)))

        curr_img = cv2.imread('{:s}/frame{:05d}.jpg'.format(src_dir, frame_id + 1))

        if len(curr_img.shape) == 2:
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)

        drawRegion(curr_img, curr_corners_selected, gt_col, gt_thickness,
                   annotate_corners=True, annotation_col=annotation_col)

        fps_text = 'frame: {:4d}'.format(frame_id + 1)
        cv2.putText(curr_img, fps_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
        cv2.imshow(gt_corners_window_name, curr_img)

        writeCorners(sel_corners_fid, curr_corners_selected, frame_id + 1)

        if hom_from_prev:
            prev_corners_gt = curr_corners_gt.copy()
            prev_corners_selected = curr_corners_selected.copy()

        key = cv2.waitKey(1 - pause_seq)
        if key == 27:
            break
        elif key == 32:
            pause_seq = 1 - pause_seq
    sel_corners_fid.close()
    cv2.destroyWindow(gt_corners_window_name)














