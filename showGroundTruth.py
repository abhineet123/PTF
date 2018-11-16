from Misc import readTrackingData
from Misc import readReinitGT
from Misc import readGT
from Misc import getParamDict
from Misc import readDistGridParams
from Misc import drawRegion
from Misc import getNormalizedUnitSquarePts
from Misc import drawGrid
from Misc import getPixValsRGB
from Misc import stackImages
from Misc import col_rgb
from Misc import getSyntheticSeqName

import sys
import cv2
import numpy as np
import os

import utility as util

if __name__ == '__main__':

    params_dict = getParamDict()
    actors = params_dict['actors']
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'
    # video_fmt = None
    video_fmt = 'avi'
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d.jpg'

    gt_col_name = 'green'
    gt_thickness = 2
    conn_col_name = 'blue'
    conn_thickness = 2

    convert_to_gs = 0
    show_stacked = 0
    stack_order = 0  # 0: row major 1: column major
    show_grid = 0
    show_patch = 0
    patch_in_border = 1
    patch_corner = 1
    resize_factor = 2
    label_corners = 0

    show_grid_label_connector = 0
    conn_end_pt_1 = (375, 0)
    conn_end_pt_2 = (375, 0)
    conn_all_corners = 1
    conn_boundary = 1

    grid_res_x = 50
    grid_res_y = 50
    grid_sub_sampling = 10

    opt_gt_ssm = '0'
    use_opt_gt = 1
    use_reinit_gt = 0

    actor_id = 5
    seq_id = 0
    seq_ids = None
    # seq_ids = [0, 1, 2, 3, 16]
    init_frame_id = 0

    show_all_seq = 1
    start_id = 0
    end_id = -1
    pause_seq = 1

    write_img = 0
    show_frame_id = 0
    dst_root_dir = '../../..//206'

    show_legend = 1
    legend_font_size = 1.5
    legend_font_thickness = 2
    legend_font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
    if cv2.__version__.startswith('3'):
        legend_font_line_type = cv2.LINE_AA
    else:
        legend_font_line_type = cv2.CV_AA
    legend_bkg_col = (0, 0, 0)

    # settings for synthetic sequences
    syn_ssm = 'c8'
    syn_ssm_sigma_id = 28
    syn_ilm = 'rbf'
    syn_am_sigma_id = 9
    syn_add_noise = 1
    syn_noise_mean = 0
    syn_noise_sigma = 10
    syn_frame_id = 0
    syn_err_thresh = 5.0

    actor = actors[actor_id]
    print 'actor: ', actor
    
    sequences = sequences[actor]
    n_seq = len(sequences)

    gt_col = col_rgb[gt_col_name]
    conn_col = col_rgb[conn_col_name]

    n_pts = grid_res_x * grid_res_y
    patch_size_x = grid_res_x * resize_factor
    patch_size_y = grid_res_y * resize_factor
    ss_grid_res_x = grid_res_x / grid_sub_sampling
    ss_grid_res_y = grid_res_y / grid_sub_sampling
    ss_n_pts = ss_grid_res_x * ss_grid_res_y

    print 'ss_n_pts: ', ss_n_pts
    print 'grid_res_x: ', grid_res_x
    print 'grid_res_y: ', grid_res_y
    print 'ss_grid_res_x: ', ss_grid_res_x
    print 'ss_grid_res_y: ', ss_grid_res_y

    curr_pts_ss = np.zeros((2, ss_n_pts), dtype=np.float64)

    std_pts, std_corners = getNormalizedUnitSquarePts(grid_res_x, grid_res_y, 0.5)
    std_pts_hm = util.homogenize(std_pts)

    std_pts_ss, ss_std_corners = getNormalizedUnitSquarePts(ss_grid_res_x, ss_grid_res_y, 0.5)
    std_pts_ss_hm = util.homogenize(std_pts_ss)


    if seq_ids is None:
        if show_all_seq:
            start_id = 0
            end_id = n_seq - 1
            print 'n_seq: ', n_seq
        elif end_id < start_id:
            start_id = end_id = seq_id
        seq_ids = range(start_id, end_id + 1)

    for curr_seq_id in seq_ids:
        seq_name = sequences[curr_seq_id]
        if actor == 'Synthetic':
            seq_name = getSyntheticSeqName(seq_name, syn_ssm, syn_ssm_sigma_id, syn_ilm,
                                syn_am_sigma_id, syn_frame_id, syn_add_noise,
                                syn_noise_mean, syn_noise_sigma)
        # seq_name = 'nl_mugII_s1'

        dst_folder = '{:s}/{:s}'.format(dst_root_dir, seq_name)
        if write_img:
            print 'Writing images to: {:s}'.format(dst_folder)
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)

        print 'sequence {:d} : {:s}'.format(curr_seq_id, seq_name)

        if opt_gt_ssm == '0':
            print 'Using standard ground truth'
            use_opt_gt = 0
        else:
            use_opt_gt = 1
            print 'Using optimized ground truth with ssm: ', opt_gt_ssm

        if use_reinit_gt:
            if use_opt_gt:
                gt_corners_fname = '{:s}/{:s}/ReinitGT/{:s}_{:s}.bin'.format(db_root_dir, actor, seq_name, opt_gt_ssm)
            else:
                gt_corners_fname = '{:s}/{:s}/ReinitGT/{:s}.bin'.format(db_root_dir, actor, seq_name)
        else:
            if use_opt_gt:
                gt_corners_fname = '{:s}/{:s}/OptGT/{:s}_{:s}.txt'.format(db_root_dir, actor, seq_name, opt_gt_ssm)
            else:
                gt_corners_fname = '{:s}/{:s}/{:s}.txt'.format(db_root_dir, actor, seq_name)

        print 'Reading ground truth from:: ', gt_corners_fname
        if use_reinit_gt:
            no_of_frames, ground_truth = readReinitGT(gt_corners_fname, init_frame_id)
        else:
            no_of_frames, ground_truth = readGT(gt_corners_fname)

        src_dir = db_root_dir + '/' + actor + '/' + seq_name
        if video_fmt is None:
            src_fname = db_root_dir + '/' + actor + '/' + seq_name + '/' + img_name_fmt
            file_list = [each for each in os.listdir(src_dir) if each.endswith('.jpg')]
            n_files = len(file_list)
        else:
            src_fname = db_root_dir + '/' + actor + '/' + seq_name + '.' + video_fmt
            n_files = -1
        cap = cv2.VideoCapture()
        if not cap.open(src_fname):
            print 'The image sequence ', src_fname, ' could not be opened'
            sys.exit()

        # img_width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        # img_height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        if n_files > 0 and n_files != no_of_frames:
            raise StandardError(
                'No. of frames in the ground truth: {:d} does not match the no. of images in the sequence: {:d}'.format(
                    no_of_frames, n_files))

        print 'no_of_frames: ', no_of_frames

        gt_corners_window_name = '{:d} : {:s}'.format(curr_seq_id, seq_name)
        if init_frame_id > 0:
            print 'Skipping {:d} frames'.format(init_frame_id + 1)
            for frame_id in xrange(init_frame_id):
                ret, curr_img = cap.read()
                if not ret:
                    raise StandardError('End of sequence reached unexpectedly')
        cv2.namedWindow(gt_corners_window_name)
        init_img = None
        for frame_id in xrange(init_frame_id, no_of_frames):


            if use_reinit_gt:
                curr_corners = np.asarray(ground_truth[frame_id - init_frame_id])
            else:
                curr_corners = np.asarray(ground_truth[frame_id])
            curr_corners = np.asarray([curr_corners[0:2].tolist(),
                                       curr_corners[2:4].tolist(),
                                       curr_corners[4:6].tolist(),
                                       curr_corners[6:8].tolist()]).T
            ret, curr_img = cap.read()
            if not ret:
                raise StandardError('End of sequence reached unexpectedly')
            img_height = curr_img.shape[0]
            img_width = curr_img.shape[1]
            # print 'img_width: ', img_width
            # print 'img_height: ', img_height
            if patch_corner == 0:
                patch_start_x = 0
                patch_start_y = 0
                patch_end_x = patch_size_x
                patch_end_y = patch_size_y
            elif patch_corner == 1:
                patch_start_x = img_width - patch_size_x
                patch_start_y = 0
                patch_end_x = img_width
                patch_end_y = patch_size_y
            if convert_to_gs:
                if len(curr_img.shape) == 3:
                    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
            n_channels = 3
            if len(curr_img.shape) == 2:
                curr_img = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)

            if np.isnan(curr_corners).any():
                print 'curr_corners:\n', curr_corners
                raise StandardError('Gound truth for frame {:d} contains NaN'.format(frame_id))

            curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners))
            curr_pts = util.dehomogenize(curr_hom_mat * std_pts_hm)
            curr_pts_ss = util.dehomogenize(curr_hom_mat * std_pts_ss_hm)

            if show_patch:
                curr_pixel_vals = getPixValsRGB(curr_pts, curr_img.astype(np.float64))
                # print 'curr_pixel_vals: ', curr_pixel_vals
                n_channels = curr_pixel_vals.shape[1]
                for channel_id in xrange(n_channels):
                    curr_patch = np.reshape(curr_pixel_vals[:, channel_id], (grid_res_y, grid_res_x)).astype(np.uint8)
                    # print 'curr_patch: ', curr_patch
                    curr_patch_resized = cv2.resize(curr_patch, (patch_size_x, patch_size_y))
                    try:
                        curr_img[patch_start_y:patch_end_y, patch_start_x:patch_end_x, channel_id] = curr_patch_resized
                    except IndexError:
                        curr_img[patch_start_y:patch_end_y, patch_start_x:patch_end_x] = curr_patch_resized
            else:
                if show_frame_id:
                    fps_text = 'frame: {:4d}'.format(frame_id + 1)
                    cv2.putText(curr_img, fps_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, gt_col)
            if show_grid:
                drawGrid(curr_img, curr_pts_ss, ss_grid_res_x, ss_grid_res_y, gt_col, gt_thickness)
            else:
                drawRegion(curr_img, curr_corners, gt_col, gt_thickness, label_corners)

            if show_grid_label_connector:
                p1 = (int(curr_corners[0, 0]), int(curr_corners[1, 0]))
                p2 = (int(curr_corners[0, 2]), int(curr_corners[1, 2]))
                cv2.line(curr_img, p1, conn_end_pt_1, conn_col, conn_thickness, cv2.CV_AA)
                cv2.line(curr_img, p2, conn_end_pt_2, conn_col, conn_thickness, cv2.CV_AA)
                if conn_all_corners:
                    p1 = (int(curr_corners[0, 1]), int(curr_corners[1, 1]))
                    p2 = (int(curr_corners[0, 3]), int(curr_corners[1, 3]))
                    cv2.line(curr_img, p1, conn_end_pt_1, conn_col, conn_thickness, cv2.CV_AA)
                    cv2.line(curr_img, p2, conn_end_pt_2, conn_col, conn_thickness, cv2.CV_AA)
                if conn_boundary:
                    drawRegion(curr_img, curr_corners, conn_col, conn_thickness, False)

            if show_stacked:
                if frame_id == init_frame_id:
                    init_img = curr_img.copy()
                displayed_img = stackImages([init_img, curr_img], stack_order)
            else:
                displayed_img = curr_img
            if show_legend:
                cv2.putText(displayed_img, 'frame {:d}'.format(frame_id + 1), (10, 20),
                            legend_font_face, legend_font_size, (0, 0, 255), 2, legend_font_line_type)
            cv2.imshow(gt_corners_window_name, displayed_img)
            if write_img:
                out_fname = '{:s}/{:s}_frame{:05d}.jpg'.format(dst_folder, seq_name, frame_id + 1)
                cv2.imwrite(out_fname, displayed_img)

            key = cv2.waitKey(1 - pause_seq)
            if key == 27:
                break
            elif key == 32:
                pause_seq = 1 - pause_seq
        cv2.destroyWindow(gt_corners_window_name)














