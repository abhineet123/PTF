from Misc import readTrackingData
from Misc import getParamDict
from Misc import readDistGridParams
from Misc import drawRegion
from Misc import writeCorners
from Misc import getTrackingObject2

import sys
import cv2
import numpy as np
import os
import glob
import math

import utility as util

curr_img = None
drawn_img = None

curr_corners = None
curr_pts = None
curr_patch = None
curr_patch_resized = None

curr_pt_id = 0

left_click_location = None
obj_selected_mouse = False
obj_selected_kb = False

final_location = None
final_location_selected_mouse = True
final_location_selected_kb = True
title = None

if __name__ == '__main__':

    params_dict = getParamDict()
    # param_ids = readDistGridParams()
    pause_seq = 0
    gt_col = (0, 255, 0)
    actors = params_dict['actors']
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d'
    img_name_ext = 'jpg'
    res_x = 50
    res_y = 50
    n_pts = res_x * res_y
    resize_factor = 4
    trans_unit = 0.1
    show_error = 1
    show_patches = 1
    show_error = show_error and show_patches

    patch_size_x = res_x * resize_factor
    patch_size_y = res_y * resize_factor
    # actor_id = param_ids['actor_id']
    # seq_id = param_ids['seq_id']
    # init_frame_id = param_ids['inc_id']

    actor = 'DFT'
    # sequences = sequences[actor]
    seq_name = 'experimental_setup_moving_camera'
    init_frame_id = 0

    from_last_frame = False
    manual_init = False

    print 'actor: ', actor
    print 'seq_name: ', seq_name

    img_name_fmt = '{:s}.{:s}'.format(img_name_fmt, img_name_ext)
    src_dir = db_root_dir + '/' + actor + '/' + seq_name
    src_fname = src_dir + '/' + img_name_fmt

    no_of_frames = len(glob.glob1(src_dir, '*.{:s}'.format(img_name_ext)))
    # no_of_frames = len([name for name in os.listdir(src_dir) if os.path.isfile(name)])
    print 'no_of_frames in source directory: ', no_of_frames

    # cap = cv2.VideoCapture()
    # if not cap.open(src_fname):
    # print 'The video file ', src_fname, ' could not be opened'
    # sys.exit()
    # ret, init_img = cap.read()

    fname = '{:s}/frame{:05d}.jpg'.format(src_dir, init_frame_id + 1)
    print 'fname: ', fname

    init_img = cv2.imread(fname)
    if len(init_img.shape) == 2:
        init_img = cv2.cvtColor(init_img, cv2.COLOR_GRAY2RGB)
    gt_corners_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'

    pose_fname = db_root_dir + '/' + actor + '/' + seq_name + '/poseGroundtruth_new.txt'
    calibration_fname = db_root_dir + '/' + actor + '/' + seq_name + '/internalCalibrationMatrix_new.txt'
    pose_data = np.loadtxt(pose_fname, dtype=np.float64, delimiter='\t')
    calibration_matrix = np.mat(np.loadtxt(calibration_fname, dtype=np.float64, delimiter='\t'))

    print 'calibration_matrix:', calibration_matrix

    if pose_data.shape[1] != no_of_frames:
        raise StandardError('Size of pose data does not match no. of frames in the sequence')

    sel_pts = getTrackingObject2(init_img)
    init_corners = np.asarray(sel_pts).astype(np.float64).T
    print 'init_corners:', init_corners
    init_corners_hm = np.mat(util.homogenize(init_corners))
    print 'init_corners_hm:', init_corners_hm

    from_last_frame = True
    pause_seq = True
    gt_frames = no_of_frames

    init_rot_mat = np.mat(np.reshape(pose_data[:9, 0], (3, 3)).transpose())
    init_trans_mat = np.mat(pose_data[9:, 0])
    init_trans_mat = init_trans_mat.transpose()
    print 'init_rot_mat:\n', init_rot_mat
    print 'init_trans_mat\n:', init_trans_mat

    init_trans_mat_tiled = np.tile(init_trans_mat, 4)
    print 'init_trans_mat_tiled\n:', init_trans_mat_tiled

    init_corners_3d = np.mat(
        np.linalg.inv(init_rot_mat) * (np.linalg.inv(calibration_matrix) * init_corners_hm - init_trans_mat_tiled))
    print 'init_corners_3d:', init_corners_3d

    gt_corners_window_name = seq_name
    cv2.namedWindow(gt_corners_window_name)

    frame_id = 1

    while frame_id < no_of_frames:
        curr_rot_mat = np.mat(np.reshape(pose_data[:9, frame_id], (3, 3)).transpose())
        curr_trans_mat = np.mat(pose_data[9:, frame_id])
        curr_trans_mat = curr_trans_mat.transpose()

        print 'curr_rot_mat:\n', curr_rot_mat
        print 'curr_trans_mat:\n', curr_trans_mat

        curr_trans_mat_tiled = np.tile(curr_trans_mat, 4)
        print 'curr_trans_mat_tiled:\n', curr_trans_mat_tiled

        curr_corners_hm = calibration_matrix * (curr_rot_mat * init_corners_3d + curr_trans_mat)
        curr_corners = util.dehomogenize(curr_corners_hm)

        curr_img = cv2.imread('{:s}/frame{:05d}.jpg'.format(src_dir, frame_id))
        drawRegion(curr_img, curr_corners, gt_col, 1)
        frame_id += 1

        cv2.imshow(gt_corners_window_name, curr_img)
        key = cv2.waitKey(1)
        if key == 32 or key == ord('p'):
            pause_seq = not pause_seq
        elif key == 27:
            break
            # corrected_gt.append(curr_corners)
            # n_corrected_get = len(corrected_gt)
            # out_file = open(corrected_corners_fname, 'w')
            # out_file.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
            # for i in xrange(n_corrected_get):
            # writeCorners(out_file, corrected_gt[i], i + 1)
            # out_file.close()














