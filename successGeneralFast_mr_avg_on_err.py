from Misc import getParamDict
from Misc import readDistGridParams
from Misc import readReinitGT
from Misc import readGT
from Misc import getSyntheticSeqName
from Misc import getSyntheticSeqSuffix

from Misc import getBinaryPtsImage2
import itertools as it

import numpy as np
import math
import sys
import os
import zipfile
import cv2

from Misc import col_rgb


def getMeanCornerDistanceError(tracker_pos, gt_pos, _overflow_err=1e3):
    # mean corner distance error
    err = 0
    for corner_id in xrange(4):
        try:
            # err += math.sqrt(
            # (float(tracking_data_line[2 * corner_id + 1]) - float(gt_line[2 * corner_id + 1])) ** 2
            # + (float(tracking_data_line[2 * corner_id + 2]) - float(gt_line[2 * corner_id + 2])) ** 2
            # )
            err += math.sqrt(
                (tracker_pos[2 * corner_id] - gt_pos[2 * corner_id]) ** 2
                + (tracker_pos[2 * corner_id + 1] - gt_pos[2 * corner_id + 1]) ** 2
            )
        except OverflowError:
            err += _overflow_err
            continue
    err /= 4.0
    # for corner_id in range(1, 9):
    # try:
    # err += (float(tracking_data_line[corner_id]) - float(gt_line[corner_id])) ** 2
    # except OverflowError:
    # continue
    # err = math.sqrt(err / 4)
    return err


def getCenterLocationError(tracker_pos, gt_pos):
    # center location error
    # centroid_tracker_x = (float(tracking_data_line[1]) + float(tracking_data_line[3]) + float(
    # tracking_data_line[5]) + float(tracking_data_line[7])) / 4.0
    # centroid2_x = (float(gt_line[1]) + float(gt_line[3]) + float(gt_line[5]) + float(
    # gt_line[7])) / 4.0

    centroid_tracker_x = (tracker_pos[0] + tracker_pos[2] + tracker_pos[4] +
                          tracker_pos[6]) / 4.0
    centroid_gt_x = (gt_pos[0] + gt_pos[2] + gt_pos[4] + gt_pos[6]) / 4.0

    # centroid_tracker_y = (float(tracking_data_line[2]) + float(tracking_data_line[4]) + float(
    # tracking_data_line[6]) + float(tracking_data_line[8])) / 4.0
    # centroid2_y = (float(gt_line[2]) + float(gt_line[4]) + float(gt_line[6]) + float(
    # gt_line[8])) / 4.0

    centroid_tracker_y = (tracker_pos[1] + tracker_pos[3] + tracker_pos[5] +
                          tracker_pos[7]) / 4.0
    centroid_gt_y = (gt_pos[1] + gt_pos[3] + gt_pos[5] + gt_pos[7]) / 4.0

    err = math.sqrt((centroid_tracker_x - centroid_gt_x) ** 2 + (centroid_tracker_y - centroid_gt_y) ** 2)
    # print 'tracking_data_line: ', tracking_data_line
    # print 'gt_line: ', gt_line
    # print 'centroid1_x: {:15.9f} centroid1_y:  {:15.9f}'.format(centroid1_x, centroid1_y)
    # print 'centroid2_x: {:15.9f} centroid2_y:  {:15.9f}'.format(centroid2_x, centroid2_y)
    # print 'err: {:15.9f}'.format(err)

    return err


def getJaccardError(tracker_pos, gt_pos, show_img=0, border_size=100, min_thresh=0, max_thresh=2000):
    min_x = int(min([tracker_pos[0], tracker_pos[2], tracker_pos[4], tracker_pos[6],
                     gt_pos[0], gt_pos[2], gt_pos[4], gt_pos[6]]))
    min_y = int(min([tracker_pos[1], tracker_pos[3], tracker_pos[5], tracker_pos[7],
                     gt_pos[1], gt_pos[3], gt_pos[5], gt_pos[7]]))
    max_x = int(max([tracker_pos[0], tracker_pos[2], tracker_pos[4], tracker_pos[6],
                     gt_pos[0], gt_pos[2], gt_pos[4], gt_pos[6]]))
    max_y = int(max([tracker_pos[1], tracker_pos[3], tracker_pos[5], tracker_pos[7],
                     gt_pos[1], gt_pos[3], gt_pos[5], gt_pos[7]]))

    if min_x < min_thresh:
        min_x = min_thresh
    if min_y < min_thresh:
        min_y = min_thresh
    if max_x > max_thresh:
        max_x = max_thresh
    if max_y > max_thresh:
        max_y = max_thresh

    if min_x > max_x or min_y > max_y:
        print 'tracker_pos: ', tracker_pos
        print 'gt_pos: ', gt_pos
        raise StandardError('Invalid Tracker and/or GT position')

    img_size = (max_y - min_y + 2 * border_size + 1, max_x - min_x + 2 * border_size + 1)

    tracker_pos_pts = np.asarray(
        [[tracker_pos[0] + border_size - min_x, tracker_pos[2] + border_size - min_x,
          tracker_pos[4] + border_size - min_x, tracker_pos[6] + border_size - min_x],
         [tracker_pos[1] + border_size - min_y, tracker_pos[3] + border_size - min_y,
          tracker_pos[5] + border_size - min_y, tracker_pos[7] + border_size - min_y]]
    )
    gt_pos_pts = np.asarray(
        [[gt_pos[0] + border_size - min_x, gt_pos[2] + border_size - min_x,
          gt_pos[4] + border_size - min_x, gt_pos[6] + border_size - min_x],
         [gt_pos[1] + border_size - min_y, gt_pos[3] + border_size - min_y,
          gt_pos[5] + border_size - min_y, gt_pos[7] + border_size - min_y]]
    )

    tracker_img = getBinaryPtsImage2(img_size, tracker_pos_pts)
    gt_img = getBinaryPtsImage2(img_size, gt_pos_pts)

    intersection_img = cv2.bitwise_and(tracker_img, gt_img)
    union_img = cv2.bitwise_or(tracker_img, gt_img)
    n_intersectio_pix = np.sum(intersection_img)
    n_union_pix = np.sum(union_img)
    jacc_error = 1.0 - float(n_intersectio_pix) / float(n_union_pix)

    if show_img:
        legend_font_size = 1
        legend_font_thickness = 1
        legend_font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
        legend_font_line_type = cv2.CV_AA
        header_location = (0, 20)

        cv2.putText(tracker_img, '{:f}'.format(jacc_error), header_location, legend_font_face,
                    legend_font_size, col_rgb['white'], legend_font_thickness, legend_font_line_type)
        cv2.putText(intersection_img, '{:d}'.format(n_intersectio_pix), header_location, legend_font_face,
                    legend_font_size, col_rgb['white'], legend_font_thickness, legend_font_line_type)
        cv2.putText(union_img, '{:d}'.format(n_union_pix), header_location, legend_font_face,
                    legend_font_size, col_rgb['white'], legend_font_thickness, legend_font_line_type)
        cv2.imshow('tracker_img', tracker_img)
        cv2.imshow('gt_img', gt_img)
        cv2.imshow('intersection_img', intersection_img)
        cv2.imshow('union_img', union_img)

        if cv2.waitKey(1) == 27:
            sys.exit(0)
    return jacc_error


def getTrackingErrors(tracker_path_orig, gt_path, _arch_fid=None, _reinit_from_gt=0,
                      _reinit_frame_skip=5, _use_reinit_gt=0, start_ids=None, _err_type=0,
                      _overflow_err=1e3, _show_jaccard_img=0):
    print 'Reading ground truth from: {:s}...'.format(gt_path)
    if _use_reinit_gt:
        n_gt_frames, gt_data = readReinitGT(gt_path, 0)
    else:
        n_gt_frames, gt_data = readGT(gt_path)

    if n_gt_frames is None or gt_data is None:
        print "Ground truth could not be read successfully"
        return None, None

    if start_ids is None:
        start_ids = [0]

    tracking_errors = []
    failure_count = 0
    for start_id in start_ids:
        if start_id == 0:
            tracker_path = tracker_path_orig
        else:
            tracker_path = tracker_path_orig.replace('.txt', '_init_{:d}.txt'.format(start_id))
        print 'Reading tracking data for start_id {:d} from: {:s}...'.format(start_id, tracker_path)
        if _arch_fid is not None:
            tracking_data = _arch_fid.open(tracker_path, 'r').readlines()
        else:
            tracking_data = open(tracker_path, 'r').readlines()
        if len(tracking_data) < 2:
            print 'Tracking data file is invalid.'
            return None, None
        # remove header
        del (tracking_data[0])
        n_lines = len(tracking_data)

        if not _reinit_from_gt and n_lines != n_gt_frames - start_id:
            print "No. of frames in tracking result ({:d}) and the ground truth ({:d}) do not match".format(
                n_lines, n_gt_frames)
            return None, None

        reinit_gt_id = 0
        reinit_start_id = 0
        # ignore the first frame where tracker was initialized
        line_id = 1
        invalid_tracker_state_found = False
        is_initialized = True
        # id of the last frame where tracking failure was detected
        failure_frame_id = -1

        while line_id < n_lines:
            tracking_data_line = tracking_data[line_id].strip().split()
            frame_fname = str(tracking_data_line[0])
            fname_len = len(frame_fname)
            frame_fname_1 = frame_fname[0:5]
            frame_fname_2 = frame_fname[- 4:]
            if frame_fname_1 != 'frame' or frame_fname_2 != '.jpg':
                print 'Invaid formatting on tracking data line {:d}: {:s}'.format(line_id + 1, tracking_data_line)
                print 'frame_fname: {:s} fname_len: {:d} frame_fname_1: {:s} frame_fname_2: {:s}'.format(
                    frame_fname, fname_len, frame_fname_1, frame_fname_2)
                return None, None
            frame_id_str = frame_fname[5:-4]
            frame_num = int(frame_id_str)
            if not _reinit_from_gt and frame_num != start_id + line_id + 1:
                print "Unexpected frame number {:d} found in line {:d} of tracking result for start_id {:d}: {:s}".format(
                    frame_num, line_id + 1, start_id, tracking_data_line)
                return None, None
            if is_initialized:
                # id of the frame in which the tracker is reinitialized
                reinit_start_id = frame_num - 2
                if failure_frame_id >= 0 and reinit_start_id != failure_frame_id + _reinit_frame_skip:
                    print 'Tracker was reinitialized in frame {:d} rather than {:d} where it should have been with {:d} frames being skipped'.format(
                        reinit_start_id + 1, failure_frame_id + _reinit_frame_skip + 1, _reinit_frame_skip
                    )
                    return None, None
                is_initialized = False

            # print 'line_id: {:d} frame_id_str: {:s} frame_num: {:d}'.format(
            # line_id, frame_id_str, frame_num)
            if len(tracking_data_line) != 9:
                if _reinit_from_gt and len(tracking_data_line) == 2 and tracking_data_line[1] == 'tracker_failed':
                    print 'tracking failure detected in frame: {:d} at line {:d}'.format(frame_num, line_id + 1)
                    failure_count += 1
                    failure_frame_id = frame_num - 1
                    # skip the frame where the tracker failed as well as the one where it was reinitialized
                    # whose result will (or should) be in the line following this one
                    line_id += 2
                    is_initialized = True
                    continue
                elif len(tracking_data_line) == 2 and tracking_data_line[1] == 'invalid_tracker_state':
                    if not invalid_tracker_state_found:
                        print 'invalid tracker state detected in frame: {:d} at line {:d}'.format(frame_num,
                                                                                                  line_id + 1)
                        invalid_tracker_state_found = True
                    line_id += 1
                    tracking_errors.append(_overflow_err)
                    continue
                else:
                    print 'Invalid formatting on line {:d}: {:s}'.format(line_id, tracking_data[line_id])
                    return None, None
            # if is_initialized:frame_num
            # is_initialized = False
            # line_id += 1
            # continue


            if _use_reinit_gt:
                if reinit_gt_id != reinit_start_id:
                    n_gt_frames, gt_data = readReinitGT(gt_path, reinit_start_id)
                    reinit_gt_id = reinit_start_id
                curr_gt = gt_data[frame_num - reinit_start_id - 1]
            else:
                curr_gt = gt_data[frame_num - 1]

            curr_tracking_data = [float(tracking_data_line[1]), float(tracking_data_line[2]),
                                  float(tracking_data_line[3]), float(tracking_data_line[4]),
                                  float(tracking_data_line[5]), float(tracking_data_line[6]),
                                  float(tracking_data_line[7]), float(tracking_data_line[8])]
            # print 'line_id: {:d} gt: {:s}'.format(line_id, gt_line)

            if _err_type == 0:
                err = getMeanCornerDistanceError(curr_tracking_data, curr_gt, _overflow_err)
            elif _err_type == 1:
                err = getCenterLocationError(curr_tracking_data, curr_gt)
            elif _err_type == 2:
                err = getJaccardError(curr_tracking_data, curr_gt, _show_jaccard_img)
            else:
                print 'Invalid error type provided: {:d}'.format(_err_type)
                return None, None
            tracking_errors.append(err)
            line_id += 1
        if _reinit_from_gt and n_lines < n_gt_frames - failure_count * (_reinit_frame_skip - 1):
            print "Unexpected no. of frames in reinit tracking result ({:d}) which should be at least {:d}".format(
                n_lines, n_gt_frames - failure_count * (_reinit_frame_skip - 1))
            return None, None
    return tracking_errors, failure_count


if __name__ == '__main__':
    use_arch = 1
    arch_root_dir = './C++/MTF/log/archives'
    arch_name = 'ncc__esm__3_4_6__lintrack'
    in_arch_path = 'tracking_data'
    tracking_root_dir = './C++/MTF/log/tracking_data'
    gt_root_dir = '../Datasets'
    out_dir = './C++/MTF/log/success_rates'
    err_out_dir = './C++/MTF/log/tracking_errors'
    # set this to '0' to use the original ground truth
    mtf_sm = 'ialkC1'
    mtf_am = 'rscv50r30i4u'
    mtf_ssm = '8'
    iiw = 1
    opt_gt_ssm = '0'
    use_reinit_gt = 1
    write_to_bin = 0
    n_runs = 1

    reinit_on_failure = 0
    reinit_frame_skip = 5
    reinit_err_thresh = 20
    reinit_at_each_frame = 0
    # mtf_am = 'miIN50r10i8b'
    # mtf_am = 'miIN50r10i1i5000s'
    enable_subseq = 0
    n_subseq = 10

    err_min = 0
    err_max = 20.0
    err_res = 100
    err_type = 0

    overflow_err = 1e3
    write_err = 1

    show_jaccard_img = 0

    # settings for synthetic sequences
    syn_ssm = 'c8'
    syn_ssm_sigma_id = 19
    syn_ilm = '0'
    syn_am_sigma_id = 0
    syn_add_noise = 0
    syn_noise_mean = 0
    syn_noise_sigma = 10
    syn_frame_id = 0

    params_dict = getParamDict()
    param_ids = readDistGridParams()

    actors = params_dict['actors']
    sequences = params_dict['sequences']
    mtf_sms = params_dict['mtf_sms']
    mtf_ams = params_dict['mtf_ams']
    mtf_ssms = params_dict['mtf_ssms']

    actor_id = param_ids['actor_id']
    mtf_sm_id = param_ids['mtf_sm_id']
    mtf_am_id = param_ids['mtf_am_id']
    mtf_ssm_id = param_ids['mtf_ssm_id']

    overriding_seq_id = -1
    # iiw = param_ids['init_identity_warp']

    arg_id = 1
    if len(sys.argv) > arg_id:
        actor_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        mtf_sm = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        mtf_am = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        mtf_ssm = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        iiw = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        arch_name = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        in_arch_path = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        arch_root_dir = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        gt_root_dir = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        tracking_root_dir = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        out_dir = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        opt_gt_ssm = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        use_reinit_gt = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        reinit_on_failure = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        reinit_frame_skip = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        reinit_err_thresh = float(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        reinit_at_each_frame = float(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        enable_subseq = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        n_subseq = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        overriding_seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        err_min = float(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        err_max = float(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        err_res = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        err_type = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        write_err = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        overflow_err = float(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        write_to_bin = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        n_runs = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_ssm = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_ssm_sigma_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_ilm = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_am_sigma_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_add_noise = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_noise_mean = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_noise_sigma = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_frame_id = int(sys.argv[arg_id])
        arg_id += 1


    use_opt_gt = 0

    if opt_gt_ssm == '0':
        print 'Using standard ground truth'
        use_opt_gt = 0
    else:
        use_opt_gt = 1
        print 'Using optimized ground truth with ssm: ', opt_gt_ssm
        if opt_gt_ssm == '2' or opt_gt_ssm == '2r':
            print 'Using center location error'

    if mtf_sm is None:
        mtf_sm = mtf_sms[mtf_sm_id]
    if mtf_am is None:
        mtf_am = mtf_ams[mtf_am_id]
    if mtf_ssm is None:
        mtf_ssm = mtf_ssms[mtf_ssm_id]
    actor = actors[actor_id]
    sequences = sequences[actor]

    using_pf = False
    if mtf_sm == 'pf':
        using_pf = True

    # err_thresholds = range(1, 20)
    err_thresholds = np.linspace(err_min, err_max, err_res)

    if overriding_seq_id >= 0:
        print 'Generating results for only sequence {:d}: {:s}'.format(
            overriding_seq_id, sequences[overriding_seq_id])
        seq_ids = [overriding_seq_id]
    else:
        seq_ids = range(0, len(sequences))
    # seq_ids = range(2, 3)
    n_err_thr = err_thresholds.size

    n_seq = len(seq_ids)
    # if actor_id == 0:
    # n_seq = 98

    if err_type == 0:
        err_name = 'Mean Corner Distance'
        err_postfix = 'mcd'
    elif err_type == 1:
        err_name = 'Center Location'
        err_postfix = 'cl'
    elif err_type == 2:
        err_name = 'Jaccard'
        err_postfix = 'jaccard'
    else:
        raise StandardError('Invalid error type provided: {:d}'.format(err_type))

    print 'Using {:s} error to evaluate tracking performance'.format(err_name)

    if n_runs > 1:
        print 'Average trackking errors over {:d} runs'.format(n_runs)

    print 'actor: {:s}'.format(actor)
    print 'mtf_sm: {:s}'.format(mtf_sm)
    print 'mtf_am: {:s}'.format(mtf_am)
    print 'mtf_ssm: {:s}'.format(mtf_ssm)
    print 'iiw: {:d}'.format(iiw)
    print 'use_opt_gt: {:d}'.format(use_opt_gt)
    print 'n_seq: {:d}'.format(n_seq)

    # if os.path.isfile(out_path):
    # s = raw_input('Output file already exists:\n {:s}\n Proceed with overwrite ?\n'.format(out_path))
    # if s == 'n' or s == 'N':
    # sys.exit()

    start_ids = None
    subseq_start_ids = None
    if enable_subseq:
        subseq_file_path = '{:s}/{:s}/subseq_start_ids_{:d}.txt'.format(gt_root_dir, actor, n_subseq)
        print 'Reading sub sequence start frame ids from : {:s}'.format(subseq_file_path)
        subseq_start_ids = np.loadtxt(subseq_file_path, delimiter=',', dtype=np.uint32)
        if subseq_start_ids.shape[0] != len(sequences) or subseq_start_ids.shape[1] != n_subseq:
            raise StandardError('Sub sequence start id file has invalid data size: {:d} x {:d}'.format(
                subseq_start_ids.shape[0], subseq_start_ids.shape[1]))

    arch_fid = None
    if use_arch:
        arch_path = '{:s}/{:s}.zip'.format(arch_root_dir, arch_name)
        print 'Reading tracking data from zip archive: {:s}'.format(arch_path)
        arch_fid = zipfile.ZipFile(arch_path, 'r')

    success_rates = np.zeros((n_err_thr, n_seq + 2), dtype=np.float64)
    success_rates[:, 0] = err_thresholds
    if use_reinit_gt:
        gt_dir = '{:s}/{:s}/ReinitGT'.format(gt_root_dir, actor)
    else:
        if use_opt_gt:
            print 'Using {:s} SSM GT'.format(opt_gt_ssm)
            gt_dir = '{:s}/{:s}/OptGT'.format(gt_root_dir, actor)
        else:
            gt_dir = '{:s}/{:s}'.format(gt_root_dir, actor)
    print 'Reading GT from folder : {:s}'.format(gt_dir)

    cmb_tracking_err = []

    total_frame_count = 0
    if reinit_at_each_frame:
        in_arch_path = '{:s}/reinit'.format(in_arch_path)
        tracking_root_dir = '{:s}/reinit'.format(tracking_root_dir)
        out_dir = '{:s}/reinit'.format(out_dir)
    elif reinit_on_failure:
        if reinit_err_thresh == int(reinit_err_thresh):
            in_arch_path = '{:s}/reinit_{:d}_{:d}'.format(in_arch_path, int(reinit_err_thresh), reinit_frame_skip)
            tracking_root_dir = '{:s}/reinit_{:d}_{:d}'.format(tracking_root_dir, int(reinit_err_thresh),
                                                               reinit_frame_skip)
            out_dir = '{:s}/reinit_{:d}_{:d}'.format(out_dir, int(reinit_err_thresh), reinit_frame_skip)
        else:
            in_arch_path = '{:s}/reinit_{:4.2f}_{:d}'.format(in_arch_path, reinit_err_thresh, reinit_frame_skip)
            tracking_root_dir = '{:s}/reinit_{:4.2f}_{:d}'.format(tracking_root_dir, reinit_err_thresh,
                                                                  reinit_frame_skip)
            out_dir = '{:s}/reinit_{:4.2f}_{:d}'.format(out_dir, reinit_err_thresh, reinit_frame_skip)

    cmb_tracking_err_list = []
    cmb_failure_count = np.zeros((1, n_seq + 2), dtype=np.float64)
    frames_per_failure = np.zeros((1, n_seq + 2), dtype=np.float64)
    cmb_avg_err = np.zeros((1, n_seq + 2), dtype=np.float64)

    out_dir = '{:s}/{:s}'.format(out_dir, actor)

    print 'tracking_root_dir: ', tracking_root_dir
    print 'in_arch_path: ', in_arch_path
    print 'out_dir: ', out_dir

    for j in xrange(n_seq):
        seq_id = seq_ids[j]
        seq_name = sequences[seq_id]
        if actor == 'Synthetic':
            seq_name = getSyntheticSeqName(seq_name, syn_ssm, syn_ssm_sigma_id, syn_ilm,
                                syn_am_sigma_id, syn_frame_id, syn_add_noise,
                                syn_noise_mean, syn_noise_sigma)

        if use_reinit_gt:
            if use_opt_gt:
                gt_path = '{:s}/{:s}_{:s}.bin'.format(gt_dir, seq_name, opt_gt_ssm)
            else:
                gt_path = '{:s}/{:s}.bin'.format(gt_dir, seq_name)
        else:
            if use_opt_gt:
                gt_path = '{:s}/{:s}_{:s}.txt'.format(gt_dir, seq_name, opt_gt_ssm)
            else:
                gt_path = '{:s}/{:s}.txt'.format(gt_dir, seq_name)
        if use_arch:
            tracking_res_path = '{:s}/{:s}/{:s}/{:s}_{:s}_{:s}_{:d}'.format(
                in_arch_path, actor, seq_name, mtf_sm, mtf_am, mtf_ssm, iiw)
        else:
            if using_pf:
                tracking_res_path = '{:s}/pf/{:s}'.format(
                    tracking_root_dir, seq_name)
            else:
                tracking_res_path = '{:s}/{:s}/{:s}/{:s}_{:s}_{:s}_{:d}'.format(
                    tracking_root_dir, actor, seq_name, mtf_sm, mtf_am, mtf_ssm, iiw)
        if enable_subseq:
            start_ids = subseq_start_ids[seq_id, :]
        if n_runs <= 1:
            # print 'seq_name: {:s}'.format(seq_name)
            tracking_res_path = '{:s}.txt'.format(tracking_res_path)
            if not use_arch and not os.path.isfile(tracking_res_path):
                print 'tracking result file not found for: {:s} in: \n {:s} '.format(seq_name, tracking_res_path)
                sys.exit(0)
            tracking_err, failure_count = getTrackingErrors(tracking_res_path, gt_path, arch_fid,
                                                            reinit_on_failure, reinit_frame_skip, use_reinit_gt,
                                                            start_ids, err_type, overflow_err, show_jaccard_img)
            if tracking_err is None:
                raise StandardError('Tracking error could not be computed for sequence {:d}: {:s}'.format(
                    j, seq_name))
        else:
            tracking_err = []
            failure_count = []
            for run_id in xrange(n_runs):
                curr_tracking_res_path = '{:s}_run_{:d}.txt'.format(run_id + 1, tracking_res_path)
                if not use_arch and not os.path.isfile(curr_tracking_res_path):
                    print 'tracking result file not found for run {:d} of {:s} in: \n {:s} '.format(
                        run_id + 1, seq_name, curr_tracking_res_path)
                    sys.exit(0)
                curr_tracking_err, curr_failure_count = getTrackingErrors(curr_tracking_res_path, gt_path, arch_fid,
                                                                reinit_on_failure, reinit_frame_skip, use_reinit_gt,
                                                                start_ids, err_type, overflow_err, show_jaccard_img)
                if curr_tracking_err is None:
                    raise StandardError('Tracking error could not be computed for run {:d} of sequence {:d}: {:s}'.format(
                        run_id + 1, j, seq_name))
                tracking_err.append(curr_tracking_err)
                failure_count.append(curr_failure_count)
            tracking_err = np.mean(np.asarray(tracking_err), axis=0)
            failure_count = np.mean(np.asarray(failure_count))

        total_frame_count += len(tracking_err)
        cmb_tracking_err.append(tracking_err)
        if reinit_at_each_frame:
            avg_err = float(sum(tracking_err)) / float(len(tracking_err))
            cmb_avg_err[0, j + 1] = avg_err
            cmb_tracking_err_list.extend(tracking_err)
        if reinit_on_failure:
            cmb_tracking_err_list.extend(tracking_err)
            cmb_failure_count[0, j + 1] = failure_count
            frames_per_failure[0, j + 1] = float(len(tracking_err)) / float(failure_count + 1)
            if len(tracking_err) == 0:
                print 'Tracker failed in all frames so setting the average error to the threshold'
                avg_err = reinit_err_thresh
            else:
                avg_err = float(sum(tracking_err)) / float(len(tracking_err))
            cmb_avg_err[0, j + 1] = avg_err
    print 'total_frame_count: ', total_frame_count
    if reinit_at_each_frame:
        cmb_avg_err[0, n_seq + 1] = float(sum(cmb_tracking_err_list)) / float(len(cmb_tracking_err_list))
    if reinit_on_failure:
        total_failure_count = np.sum(cmb_failure_count)
        if reinit_on_failure:
            print 'total_failure_count: {:d}'.format(int(total_failure_count))
        print 'total_error_count: ', len(cmb_tracking_err_list)
        cmb_failure_count[0, n_seq + 1] = total_failure_count
        cmb_avg_err[0, n_seq + 1] = float(sum(cmb_tracking_err_list)) / float(len(cmb_tracking_err_list))
        frames_per_failure[0, n_seq + 1] = float(total_frame_count) / float(total_failure_count + 1)

    # write tracking errors
    if write_err:
        if reinit_at_each_frame:
            actor_err_out_dir = '{:s}/reinit'.format(err_out_dir)
        elif reinit_on_failure:
            if reinit_err_thresh == int(reinit_err_thresh):
                actor_err_out_dir = '{:s}/reinit_{:d}_{:d}'.format(err_out_dir, int(reinit_err_thresh), reinit_frame_skip)
            else:
                actor_err_out_dir = '{:s}/reinit_{:4.2f}_{:d}'.format(err_out_dir, reinit_err_thresh, reinit_frame_skip)
        else:
            actor_err_out_dir = err_out_dir
        actor_err_out_dir = '{:s}/{:s}'.format(actor_err_out_dir, actor)
        if not os.path.exists(actor_err_out_dir):
            print 'Tracking error directory: {:s} does not exist. Creating it...'.format(actor_err_out_dir)
            os.makedirs(actor_err_out_dir)
        err_out_path = '{:s}/err_{:s}_{:s}_{:s}_{:d}'.format(
            actor_err_out_dir, mtf_sm, mtf_am, mtf_ssm, iiw)
        if use_opt_gt:
            err_out_path = '{:s}_{:s}'.format(err_out_path, opt_gt_ssm)
        if enable_subseq:
            err_out_path = '{:s}_subseq_{:d}'.format(err_out_path, n_subseq)
        if err_type:
            err_out_path = '{:s}_{:s}'.format(err_out_path, err_postfix)
        err_out_path = '{:s}.txt'.format(err_out_path)

        print 'Writing tracking errors to: {:s}'.format(err_out_path)
        # np.savetxt(err_out_path, np.array([np.array(err) for err in cmb_tracking_err]), fmt='%15.9f', delimiter='\t')
        cmb_tracking_err_arr = list(it.izip_longest(*cmb_tracking_err, fillvalue=-1))
        # print 'cmb_tracking_err_arr: ', cmb_tracking_err_arr
        np.savetxt(err_out_path, np.array([k for k in cmb_tracking_err_arr]), fmt='%15.9f', delimiter='\t')

    # compute success rates for different thresholds
    for i in xrange(n_err_thr):
        success_frame_count = 0
        err_thr = err_thresholds[i]
        for j in xrange(n_seq):
            seq_success_count = sum(err <= err_thr for err in cmb_tracking_err[j])
            success_frame_count += seq_success_count
            if len(cmb_tracking_err[j]) == 0:
                success_rates[i, j + 1] = 0
            else:
                success_rates[i, j + 1] = float(seq_success_count) / float(len(cmb_tracking_err[j]))
        success_rates[i, n_seq + 1] = float(success_frame_count) / float(total_frame_count)
        print 'err_thr: {:15.9f} sfc: {:6d} tfc: {:6d} sr: {:15.9f}'.format(
            err_thr, success_frame_count, total_frame_count, success_rates[i, n_seq + 1])

    # write success rates (and other results if available)
    if not os.path.exists(out_dir):
        print 'Output directory: {:s} does not exist. Creating it...'.format(out_dir)
        os.makedirs(out_dir)

    out_path = '{:s}/sr'.format(out_dir)
    if actor == 'Synthetic':
        syn_out_suffix = getSyntheticSeqSuffix(syn_ssm, syn_ssm_sigma_id, syn_ilm,syn_am_sigma_id,
                                      syn_add_noise, syn_noise_mean, syn_noise_sigma)
        out_path = '{:s}_{:s}'.format(out_path, syn_out_suffix)

    if overriding_seq_id >= 0:
        out_path = '{:s}_{:s}'.format(out_path, sequences[overriding_seq_id])

    out_path = '{:s}_{:s}_{:s}_{:s}_{:d}'.format(
        out_path, mtf_sm, mtf_am, mtf_ssm, iiw)
    if use_opt_gt:
        out_path = '{:s}_{:s}'.format(out_path, opt_gt_ssm)
    if enable_subseq:
        out_path = '{:s}_subseq_{:d}'.format(out_path, n_subseq)
    if err_type:
        out_path = '{:s}_{:s}'.format(out_path, err_postfix)
    if n_runs > 1:
        out_path = '{:s}_{:d}_runs'.format(out_path, n_runs)

    if write_to_bin:
        out_path = '{:s}.bin'.format(out_path)
    else:
        out_path = '{:s}.txt'.format(out_path)
    print 'Saving success rate data to {:s}'.format(out_path)
    if reinit_at_each_frame:
        out_data = np.vstack((success_rates, cmb_avg_err))
    elif reinit_on_failure:
        out_data = np.vstack((success_rates, cmb_avg_err, cmb_failure_count, frames_per_failure))
    else:
        out_data = success_rates
    if write_to_bin:
        out_file = open(out_path, 'wb')
        np.array(out_data.shape, dtype=np.uint32).tofile(out_file)
        out_data.astype(np.float64).tofile(out_file)
        out_file.close()
    else:
        np.savetxt(out_path, out_data, fmt='%15.9f', delimiter='\t')

    if use_arch:
        arch_fid.close()
