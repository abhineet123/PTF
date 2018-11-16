import numpy as np
import math
import sys
from Misc import getParamDict
from Misc import readDistGridParams
import os
import zipfile
import pdb


def compute_recoveries(thresh_recov, thresh_failure, mcd_error):
    '''
    thresh_recov: threshold for recovery
    thresh_failure: [min, max, increment]
    mcd_error
    '''
    failure_range = range(*thresh_failure)
    hist_recoveries = np.zeros((len(failure_range), 1))
    hist_failures = np.zeros((len(failure_range), 1))
    for x in failure_range:
        index = (x - thresh_failure[0]) / thresh_failure[2]
        # print('Current threshold failure ', x)
        i = 0
        while i < len(mcd_error):
            if mcd_error[i] > x:
                hist_failures[index] += 1
                for j in xrange(i + 1, len(mcd_error)):
                    i += 1
                    if mcd_error[j] < thresh_recov:
                        hist_recoveries[index] += 1
                        break
            i += 1

        # if hist_failures[index] == 0:
        #     hist_recoveries[index] = -1
        # else:
        #     hist_recoveries[index] /= hist_failures[index]

    return hist_recoveries, hist_failures


def MCDError(tracker_path, gt_path, _use_cle=0, _arch_fid=None, _reinit_from_gt=0,
             overflow_err=1e3):
    print 'Reading tracking data from: {:s}...'.format(tracker_path)
    if _arch_fid is not None:
        tracking_data = _arch_fid.open(tracker_path, 'r').readlines()
    else:
        tracking_data = open(tracker_path, 'r').readlines()
    if len(tracking_data) < 2:
        print 'Tracking data file is invalid.'
        return None, None

    print 'Reading ground truth from: {:s}...'.format(gt_path)
    gt_data = open(gt_path, 'r').readlines()
    if len(gt_data) < 2:
        print 'Tracking data file is invalid'
        return None, None

    del (tracking_data[0])
    del (gt_data[0])

    n_lines = len(tracking_data)
    n_gt_frames = len(gt_data)

    if not _reinit_from_gt and n_lines != n_gt_frames:
        print "No. of frames in tracking result ({:d}) and the ground truth ({:d}) do not match".format(
            n_lines, n_gt_frames)
        return None, None
    line_id = 1
    failure_count = 0
    mcd_err = []
    invalid_tracker_state_found = False
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
        # print 'line_id: {:d} frame_id_str: {:s} frame_num: {:d}'.format(
        # line_id, frame_id_str, frame_num)
        if len(tracking_data_line) != 9:
            if _reinit_from_gt and len(tracking_data_line) == 2 and tracking_data_line[1] == 'tracker_failed':
                print 'tracking failure detected in frame: {:d} at line {:d}'.format(frame_num, line_id + 1)
                failure_count += 1
                # skip the frame where the tracker failed as well as the one where it was reinitialized
                is_initialized = True
                line_id += 2
                continue
            elif len(tracking_data_line) == 2 and tracking_data_line[1] == 'invalid_tracker_state':
                if not invalid_tracker_state_found:
                    print 'invalid tracker state detected in frame: {:d} at line {:d}'.format(frame_num, line_id + 1)
                    invalid_tracker_state_found = True
                line_id += 1
                mcd_err.append(overflow_err)
                continue
            else:
                print 'Invalid formatting on line {:d}: {:s}'.format(line_id, tracking_data[line_id])
                return None, None
        # if is_initialized:frame_num
        # is_initialized = False
        # line_id += 1
        # continue
        gt_line = gt_data[frame_num - 1].strip().split()
        gt_frame_fname = gt_line[0]
        gt_frame_num = int(gt_frame_fname[5:-4])
        gt_frame_fname_1 = gt_frame_fname[0:5]
        gt_frame_fname_2 = gt_frame_fname[- 4:]
        if gt_frame_fname_1 != 'frame' or gt_frame_fname_2 != '.jpg' or gt_frame_num != frame_num:
            print 'Invaid formatting on GT data line {:d}: {:s}'.format(frame_num, gt_line)
            print 'gt_frame_fname_1: {:s}'.format(gt_frame_fname_1)
            print 'gt_frame_fname_2: {:s}'.format(gt_frame_fname_2)
            print 'gt_frame_num: {:d}'.format(gt_frame_num)
            return None, None
        # print 'line_id: {:d} gt: {:s}'.format(line_id, gt_line)
        if _use_cle:
            # center location error
            centroid1_x = (float(tracking_data_line[1]) + float(tracking_data_line[3]) + float(
                tracking_data_line[5]) + float(tracking_data_line[7])) / 4.0
            centroid2_x = (float(gt_line[1]) + float(gt_line[3]) + float(gt_line[5]) + float(
                gt_line[7])) / 4.0

            centroid1_y = (float(tracking_data_line[2]) + float(tracking_data_line[4]) + float(
                tracking_data_line[6]) + float(tracking_data_line[8])) / 4.0
            centroid2_y = (float(gt_line[2]) + float(gt_line[4]) + float(gt_line[6]) + float(
                gt_line[8])) / 4.0

            err = math.sqrt((centroid1_x - centroid2_x) ** 2 + (centroid1_y - centroid2_y) ** 2)
            # print 'tracking_data_line: ', tracking_data_line
            # print 'gt_line: ', gt_line
            # print 'centroid1_x: {:15.9f} centroid1_y:  {:15.9f}'.format(centroid1_x, centroid1_y)
            # print 'centroid2_x: {:15.9f} centroid2_y:  {:15.9f}'.format(centroid2_x, centroid2_y)
            # print 'err: {:15.9f}'.format(err)
        else:
            # mean corner distance error
            err = 0
            for corner_id in range(4):
                try:
                    err += math.sqrt(
                        (float(tracking_data_line[2 * corner_id + 1]) - float(gt_line[2 * corner_id + 1])) ** 2
                        + (float(tracking_data_line[2 * corner_id + 2]) - float(gt_line[2 * corner_id + 2])) ** 2
                    )
                except OverflowError:
                    err += overflow_err
                    continue
            err /= 4.0
            # for corner_id in range(1, 9):
            # try:
            # err += (float(tracking_data_line[corner_id]) - float(gt_line[corner_id])) ** 2
            # except OverflowError:
            # continue
            # err = math.sqrt(err / 4)
        mcd_err.append(err)
        line_id += 1
    return mcd_err, failure_count


if __name__ == '__main__':
    use_arch = 1
    arch_root_dir = '..'
    arch_name = 'resl_pf_500_1d__sl3__rscv_mi_ccre_scv_ncc_ssim_spss__50r_1i'
    in_arch_path = 'tracking_data'
    tracking_root_dir = '~/mtf-regnet/log/tracking_data'
    gt_root_dir = '/usr/data/Datasets'
    out_dir = '~/mtf-regnet/log/recovery_rates'

    # set this to '0' to use the original ground truth
    mtf_sm = 'ialkC1'
    mtf_am = 'rscv50r30i4u'
    mtf_ssm = '8'
    iiw = 1
    opt_gt_ssm = '0'
    reinit_from_gt = 0
    reinit_frame_skip = 5
    reinit_err_thresh = 20
    # mtf_am = 'miIN50r10i8b'
    # mtf_am = 'miIN50r10i1i5000s'

    # for histogram of recovery rate
    thresh_failure_list = [30, 40, 1]
    thresh_recov = 20

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
        reinit_from_gt = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        reinit_frame_skip = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        reinit_err_thresh = float(sys.argv[arg_id])
        arg_id += 1

    use_cle = 0
    use_opt_gt = 0

    if opt_gt_ssm == '0':
        print 'Using standard ground truth'
        use_opt_gt = 0
    else:
        use_opt_gt = 1
        print 'Using optimized ground truth with ssm: ', opt_gt_ssm
        if opt_gt_ssm == '2' or opt_gt_ssm == '2r':
            print 'Using center location error'
            use_cle = 1

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

    seq_ids = range(0, len(sequences))
    n_seq = len(seq_ids)
    # if actor_id == 0:
    # n_seq = 98

    print 'actor: {:s}'.format(actor)
    print 'mtf_sm: {:s}'.format(mtf_sm)
    print 'mtf_am: {:s}'.format(mtf_am)
    print 'mtf_ssm: {:s}'.format(mtf_ssm)
    print 'iiw: {:d}'.format(iiw)
    print 'use_opt_gt: {:d}'.format(use_opt_gt)
    print 'use_cle: {:d}'.format(use_cle)
    print 'n_seq: {:d}'.format(n_seq)

    # if os.path.isfile(out_path):
    # s = raw_input('Output file already exists:\n {:s}\n Proceed with overwrite ?\n'.format(out_path))
    # if s == 'n' or s == 'N':
    # sys.exit()

    arch_fid = None
    if use_arch:
        arch_path = '{:s}/{:s}.zip'.format(arch_root_dir, arch_name)
        print 'Reading tracking data from zip archive: {:s}'.format(arch_path)
        arch_fid = zipfile.ZipFile(arch_path, 'r')

    if use_opt_gt:
        print 'Using {:s} SSM GT'.format(opt_gt_ssm)
        gt_dir = '{:s}/{:s}/OptGT'.format(gt_root_dir, actor)
    else:
        gt_dir = '{:s}/{:s}'.format(gt_root_dir, actor)
    print 'Reading GT from folder : {:s}'.format(gt_dir)

    cmb_recovery = []

    total_frame_count = 0
    print 'tracking_root_dir: ', tracking_root_dir
    print 'in_arch_path: ', in_arch_path
    print 'out_dir: ', out_dir
    n_failure=len(range(*thresh_failure_list))
    cmbd_hist_recoveries=np.zeros((n_failure, 1), dtype=np.int32)
    cmbd_hist_failures=np.zeros((n_failure, 1), dtype=np.int32)


    for j in xrange(n_seq):
        seq_id = seq_ids[j]
        seq_name = sequences[seq_id]
        # print 'seq_name: {:s}'.format(seq_name)
        if use_arch:
            tracking_res_path = '{:s}/{:s}/{:s}/{:s}_{:s}_{:s}_{:d}.txt'.format(
                in_arch_path, actor, seq_name, mtf_sm, mtf_am, mtf_ssm, iiw)
        else:
            if using_pf:
                tracking_res_path = '{:s}/pf/{:s}.txt'.format(
                    tracking_root_dir, seq_name)
            else:
                tracking_res_path = '{:s}/{:s}/{:s}/{:s}_{:s}_{:s}_{:d}.txt'.format(
                    tracking_root_dir, actor, seq_name, mtf_sm, mtf_am, mtf_ssm, iiw)
            if not os.path.isfile(tracking_res_path):
                print 'tracking result file not found for: {:s} in: \n {:s} '.format(seq_name, tracking_res_path)
                sys.exit(0)
        if use_opt_gt:
            gt_path = '{:s}/{:s}_{:s}.txt'.format(gt_dir, seq_name, opt_gt_ssm)
        else:
            gt_path = '{:s}/{:s}.txt'.format(gt_dir, seq_name)

        mcd_err, failure_count = MCDError(
            tracking_res_path, gt_path, use_cle, arch_fid, reinit_from_gt)
        if mcd_err is None:
            raise StandardError('MCD Error could not be computed for sequece {:d}: {:s}'.format(j, seq_name))
        hist_recoveries, hist_failures = compute_recoveries(thresh_recov, thresh_failure_list, mcd_err)
        print 'seq: {:d} total_failures: {:f} total_recoveries: {:f}'.format(seq_id, hist_failures.sum(), hist_recoveries.sum())

        cmbd_hist_recoveries += hist_recoveries
        cmbd_hist_failures += hist_failures
        total_frame_count += len(mcd_err)
    print 'total_frame_count: ', total_frame_count

    if not os.path.exists(out_dir):
        print 'Output directory: {:s} does not exist. Creating it...'.format(out_dir)
        os.makedirs(out_dir)

    if use_opt_gt:
        out_path_recoveries = '{:s}/sr_{:s}_{:s}_{:s}_{:s}_{:d}_{:s}_recoveries.txt'.format(
            out_dir, actor, mtf_sm, mtf_am, mtf_ssm, iiw, opt_gt_ssm)
    else:
        out_path_recoveries = '{:s}/sr_{:s}_{:s}_{:s}_{:s}_{:d}_recoveries.txt'.format(
            out_dir, actor, mtf_sm, mtf_am, mtf_ssm, iiw)
    out_data_recoveries = np.hstack(
        (np.asarray(range(*thresh_failure_list)).reshape(-1, 1),
         np.asarray(cmbd_hist_failures).reshape(-1, 1),
         np.asarray(cmbd_hist_recoveries).reshape(-1, 1))
    )
    np.savetxt(out_path_recoveries, out_data_recoveries, fmt='%15.9f', delimiter='\t')

    if use_arch:
        arch_fid.close()


        # trackerPath = sys.argv[1]
        # GTPath = sys.argv[2]
        # TH = sys.argv[3]
        # SuccessRate(trackerPath, GTPath, TH)

