from Misc import readTrackingData
from Misc import getParamDict
from Misc import drawRegion
from Misc import drawGrid
from Misc import col_rgb
from Misc import stackImages
from Misc import getNormalizedUnitSquarePts
import sys
import os
import cv2
import numpy as np
import zipfile
from trackingDataConfig import tracker_configs
from trackingDataConfig import out_dirs

import utility as util

if __name__ == '__main__':
    params_dict = getParamDict()

    use_arch = 1
    arch_root_dir = './C++/MTF/log/archives'
    # arch_root_dir = 'O:\UofA\Results\#Old\\tracking_data'
    # arch_root_dir = 'O:\UofA\Results\#Old\CRV'
    arch_name = 'resh_dsst_2_3_4_cmt_2_3_4_tmtfineEIH_jacc'
    in_arch_path = 'tracking_data'

    db_root_dir = '../Datasets'
    tracking_data_root_dir = 'C++/MTF/log'
    out_root_dir = 'C++/MTF/log'
    out_root_dir = '../../Reports/CRV17'

    data_config_id = 33
    n_trackers = 7

    actor_id = 8
    # start_id = 3 + 1 * 16
    start_id = 43
    end_id = -1
    # start_id = 0
    # end_id = 11
    # end_id = 9 + 1 * 16
    actor = 'Live'
    seq_name = 'usb_cam'
    sub_seq_name = ''

    write_img = 1
    show_stacked = 1
    stack_order = 0  # 0: row major 1: column major
    resize_stacked_img = 1
    resize_factor = 0.75
    pause_seq = 0
    convert_to_gs = 0
    show_header = 1
    show_legend = 1
    annotate_corners = 1
    annotation_font_size = 1
    annotation_col = None

    show_grid = 0
    grid_res_x = 40
    grid_res_y = 40

    line_thickness = 2

    failure_font_size = 1.5
    failure_font_thickness = 2
    legend_font_size = 1.5
    legend_font_thickness = 2
    legend_font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
    if cv2.__version__.startswith('3'):
        legend_font_line_type = cv2.LINE_AA
    else:
        legend_font_line_type = cv2.CV_AA
    legend_bkg_col = (0, 0, 0)
    legend_gap = 2

    header_location = (0, 20)
    header_font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
    header_col = (255, 255, 255)
    header_bkg_col = (0, 0, 0)
    header_font_size = 1.3
    header_font_thickness = 1

    actors = params_dict['actors']
    sequences = params_dict['sequences']
    # mtf_sms = params_dict['mtf_sms']
    # mtf_ams = params_dict['mtf_ams']
    # mtf_ssms = params_dict['mtf_ssms']

    read_from_vid = 0
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d.jpg'
    vid_fmt = 'avi'

    if end_id < start_id:
        end_id = start_id

    if actor_id >= 0:
        seq_ids = range(start_id, end_id + 1)
    else:
        seq_ids = range(start_id, start_id + 1)

    std_pts, std_corners = getNormalizedUnitSquarePts(grid_res_x, grid_res_y, 0.5)
    std_pts_hm = util.homogenize(std_pts)

    for seq_id in seq_ids:
        # mtf_sm_id = param_ids['mtf_sm_id']
        # mtf_am_id = param_ids['mtf_am_id']
        # mtf_ssm_id = param_ids['mtf_ssm_id']
        # init_identity_warp = param_ids['init_identity_warp']

        if actor_id >= 0:
            actor = actors[actor_id]
            seq_name = sequences[actor][seq_id]
        # mtf_sm = mtf_sms[mtf_sm_id]
        # mtf_am = mtf_ams[mtf_am_id]
        # mtf_ssm = mtf_ssms[mtf_ssm_id]

        print 'actor: ', actor
        print 'seq_name: ', seq_name

        if actor is None or seq_name is None:
            raise SyntaxError('Invalid actor and/or sequence provided')

        out_dir = out_dirs[data_config_id]
        dst_folder = '{:s}/tracking_videos/{:s}_{:s}'.format(out_root_dir, out_dir, seq_name)
        if show_stacked:
            dst_folder += '_stacked'
        if show_grid:
            dst_folder += '_grid_{:d}x{:d}'.format(grid_res_x, grid_res_y)
        if write_img:
            print 'Writing images to: {:s}'.format(dst_folder)
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)

        src_fname = db_root_dir + '/' + actor + '/' + seq_name
        if sub_seq_name is not None and sub_seq_name:
            src_fname = src_fname + '/' + sub_seq_name
        if read_from_vid:
            src_fname = src_fname + '.' + vid_fmt
        else:
            src_fname = src_fname + '/' + img_name_fmt

        cap = cv2.VideoCapture()

        if not cap.open(src_fname):
            if read_from_vid:
                print 'The video file ', src_fname, ' could not be opened'
            else:
                print 'The image sequence ', src_fname, ' could not be read'
            sys.exit()

        arch_fid = None
        # if use_arch:
        # arch_path = '{:s}/{:s}.zip'.format(arch_root_dir, arch_name)
        # print 'Reading tracking data from zip archive: {:s}'.format(arch_path)
        #     arch_fid = zipfile.ZipFile(arch_path, 'r')

        tracking_data_list = []
        trackers = tracker_configs[data_config_id]
        if n_trackers > len(trackers):
            n_trackers = len(trackers)
        for tracker_id in xrange(n_trackers):
            tracker = trackers[tracker_id]
            if 'fname' in tracker.keys():
                # entire filename can be specified too for convenience
                words = tracker['fname'].split('_')
                if len(words) != 4:
                    raise StandardError('Invalid filename provided: {:s}'.format(tracker['fname']))
                tracker['sm'] = words[0]
                tracker['am'] = words[1]
                tracker['ssm'] = words[2]
                tracker['iiw'] = int(words[3])

            print 'mtf_sm: ', tracker['sm']
            print 'mtf_am: ', tracker['am']
            print 'mtf_ssm: ', tracker['ssm']
            print 'iiw: ', tracker['iiw']

            if 'use_arch' in tracker:
                use_arch = tracker['use_arch']

            if tracker['sm'] == 'gt':
                tracking_data_fname = '{:s}/{:s}/{:s}.txt'.format(
                    db_root_dir, actor, seq_name)
                arch_fid = None
            elif tracker['sm'] == 'opt_gt':
                tracking_data_fname = '{:s}/{:s}/OptGT/{:s}_{:s}.txt'.format(
                    db_root_dir, actor, seq_name, tracker['ssm'])
                arch_fid = None
            else:
                if use_arch:
                    if 'arch_name' in tracker:
                        arch_name = tracker['arch_name']
                    if 'arch_root_dir' in tracker:
                        arch_root_dir = tracker['arch_root_dir']
                    if 'in_arch_path' in tracker:
                        in_arch_path = tracker['in_arch_path']
                    arch_path = '{:s}/{:s}.zip'.format(arch_root_dir, arch_name)
                    print 'Reading data for tracker {:d} from zip archive: {:s}'.format(tracker_id, arch_path)
                    if arch_fid:
                        arch_fid.close()
                    arch_fid = zipfile.ZipFile(arch_path, 'r')
                    tracking_data_fname = '{:s}/{:s}/{:s}/{:s}_{:s}_{:s}_{:d}.txt'.format(
                        in_arch_path, actor, seq_name, tracker['sm'], tracker['am'], tracker['ssm'],
                        tracker['iiw'])
                else:
                    tracking_data_fname = '{:s}/tracking_data/{:s}/{:s}/{:s}_{:s}_{:s}_{:d}.txt'.format(
                        tracking_data_root_dir, actor, seq_name, tracker['sm'], tracker['am'], tracker['ssm'],
                        tracker['iiw'])
            print 'Reading tracking data from:: ', tracking_data_fname
            tracking_data = readTrackingData(tracking_data_fname, arch_fid)
            tracking_data_list.append(tracking_data)

        no_of_frames = tracking_data_list[0].shape[0]
        print 'no_of_frames: ', no_of_frames

        tracking_data_win_name = seq_name
        cv2.namedWindow(tracking_data_win_name)

        for frame_id in xrange(no_of_frames):

            ret, curr_img = cap.read()
            if not ret:
                print 'End of sequence reached unexpectedly'
                break
            if convert_to_gs:
                if len(curr_img.shape) == 3:
                    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
            if len(curr_img.shape) == 2:
                curr_img = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)

            curr_img_list = []

            fps_text = '{:s}: frame {:d}'.format(seq_name, frame_id + 1)
            header_text_size, header_baseline = cv2.getTextSize(fps_text, header_font_face,
                                                                header_font_size, header_font_thickness)
            legend_x = header_location[0]


            # print 'header_text_size: ', header_text_size
            # print 'legend_x: ', legend_x


            text_size_list = []
            for tracker_id in xrange(n_trackers):
                tracker = trackers[tracker_id]
                text_size, baseline = cv2.getTextSize(tracker['legend'], legend_font_face,
                                                      legend_font_size, legend_font_thickness)
                no_stack = not show_stacked
                if 'no_stack' in tracker.keys():
                    no_stack = tracker['no_stack'] and curr_img_list

                if show_stacked:
                    if no_stack:
                        legend_x += text_size_list[-1][0] + legend_gap
                    else:
                        legend_x = header_location[0]

                if show_header:
                    legend_y = header_location[1] + text_size[1] + baseline + header_baseline + legend_gap
                else:
                    legend_y = text_size[1] + baseline + legend_gap
                # print 'legend_y: ', legend_y
                # print 'text_size: ', text_size
                # print 'baseline: ', baseline
                baseline += legend_font_thickness

                tracking_data = tracking_data_list[tracker_id]
                curr_corners = np.asarray([tracking_data[frame_id, 0:2].tolist(),
                                           tracking_data[frame_id, 2:4].tolist(),
                                           tracking_data[frame_id, 4:6].tolist(),
                                           tracking_data[frame_id, 6:8].tolist()]).T
                if np.isnan(curr_corners).any():
                    if show_stacked:
                        new_img = np.copy(curr_img)
                        cv2.putText(new_img, 'Tracker Failed', (legend_x, legend_y), legend_font_face,
                                    failure_font_size, col_rgb['red'], failure_font_thickness, legend_font_line_type)
                        curr_img_list.append(new_img)
                    else:
                        cv2.putText(curr_img, 'Tracker Failed', (curr_img.shape[0] / 2, curr_img.shape[1] / 2),
                                    legend_font_face,
                                    failure_font_size, col_rgb['red'], failure_font_thickness, legend_font_line_type)
                    continue

                if show_grid:
                    try:
                        curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners))
                    except:
                        pass
                    curr_pts = util.dehomogenize(curr_hom_mat * std_pts_hm)

                if show_stacked:
                    if no_stack:
                        new_img = curr_img_list[-1]
                    else:
                        new_img = np.copy(curr_img)
                    if show_legend and tracker['legend']:
                        cv2.rectangle(new_img, (legend_x, legend_y + baseline),
                                      (legend_x + text_size[0], legend_y - text_size[1] - baseline),
                                      legend_bkg_col, -1)

                        cv2.putText(new_img, tracker['legend'], (legend_x, legend_y), legend_font_face,
                                    legend_font_size, col_rgb[tracker['col']],
                                    legend_font_thickness, legend_font_line_type)
                    if show_grid:
                        drawGrid(new_img, curr_pts, grid_res_x, grid_res_y,
                                 col_rgb[tracker['col']], line_thickness)
                    drawRegion(new_img, curr_corners, col_rgb[tracker['col']], line_thickness,
                               annotate_corners, annotation_col, annotation_font_size)
                    if not no_stack:
                        curr_img_list.append(new_img)
                        text_size_list.append(text_size)

                else:
                    if show_legend and tracker['legend']:
                        cv2.rectangle(curr_img, (legend_x, legend_y + baseline),
                                      (legend_x + text_size[0], legend_y - text_size[1] - baseline),
                                      legend_bkg_col, -1)
                        cv2.putText(curr_img, tracker['legend'], (legend_x, legend_y), legend_font_face,
                                    legend_font_size, col_rgb[tracker['col']],
                                    legend_font_thickness, legend_font_line_type)
                    if show_grid:
                        drawGrid(curr_img, curr_pts, grid_res_x, grid_res_y,
                                 col_rgb[tracker['col']], line_thickness)
                    drawRegion(curr_img, curr_corners, col_rgb[tracker['col']], line_thickness)
                    legend_x += text_size[0] + legend_gap
                    # legend_x += len(tracker['legend']) * 20 * legend_font_size
            if show_stacked:
                displayed_img = stackImages(curr_img_list, stack_order)
                if resize_stacked_img:
                    displayed_img = cv2.resize(displayed_img, (0, 0), fx=resize_factor, fy=resize_factor)
            else:
                displayed_img = curr_img

            if show_header:
                cv2.rectangle(displayed_img, (header_location[0], header_location[1] + header_baseline),
                              (header_location[0] + header_text_size[0],
                               header_location[1] - header_text_size[1] - header_baseline),
                              header_bkg_col, -1)
                cv2.putText(displayed_img, fps_text, header_location, header_font_face, header_font_size,
                            header_col, header_font_thickness)
            cv2.imshow(tracking_data_win_name, displayed_img)
            if write_img:
                out_fname = '{:s}/{:s}_frame{:05d}.jpg'.format(dst_folder, seq_name, frame_id + 1)
                cv2.imwrite(out_fname, displayed_img)

            key = cv2.waitKey(1 - pause_seq)
            if key == 27:
                break
            elif key == 32:
                pause_seq = 1 - pause_seq
        cv2.destroyWindow(tracking_data_win_name)

