import numpy as np
import math
import sys
from Misc import getParamDict
from Misc import readDistGridParams
import os
import zipfile

'''
USAGE: python <path of tracked file> <path of ground truth>
     : Notice TH (threshold) is set at 5, you can change it if you want

OUTPUT: Success rate as a fraction of the total frames tracked
'''

def SuccessRate(trackerPath, GTPath, TH, use_cle=0, arch_fid=None):
    if arch_fid is not None:
        load_Tracker = arch_fid.open(trackerPath, 'r').readlines()
    else:
        load_Tracker = open(trackerPath, 'r').readlines()
    load_GT = open(GTPath, 'r').readlines()
    no_frames = len(load_Tracker) - 1
    gt_frames = len(load_GT) - 1

    if no_frames != gt_frames:
        print "No. of frames in tracking result ({:d}) and the ground truth ({:d}) do not match".format(
            no_frames, gt_frames)
        sys.exit(0)
    E = 0
    I = 1

    while I < no_frames:
        Tracker = load_Tracker[I].strip().split()
        GT = load_GT[I].strip().split()
        if use_cle:
            centroid1_x = (float(Tracker[1]) + float(Tracker[3]) + float(Tracker[5]) + float(Tracker[7])) / 4.0
            centroid2_x = (float(GT[1]) + float(GT[3]) + float(GT[5]) + float(GT[7])) / 4.0

            centroid1_y = (float(Tracker[2]) + float(Tracker[4]) + float(Tracker[6]) + float(Tracker[8])) / 4.0
            centroid2_y = (float(GT[2]) + float(GT[4]) + float(GT[6]) + float(GT[8])) / 4.0

            err = math.sqrt((centroid1_x - centroid2_x) ** 2 + (centroid1_y - centroid2_y) ** 2)
            # print 'Tracker: ', Tracker
            # print 'GT: ', GT
            # print 'centroid1_x: {:15.9f} centroid1_y:  {:15.9f}'.format(centroid1_x, centroid1_y)
            # print 'centroid2_x: {:15.9f} centroid2_y:  {:15.9f}'.format(centroid2_x, centroid2_y)
            # print 'err: {:15.9f}'.format(err)
        else:
            err = 0
            # Alignment error
            for p in range(1, 9):
                err += (float(Tracker[p]) - float(GT[p])) ** 2
            err = math.sqrt(err / 4)
        if err < TH: E += 1
        I += 1

    success_rate = E / float(no_frames)
    return success_rate, E, no_frames


if __name__ == '__main__':
    use_arch = 1
    arch_root_dir = './C++/MTF_LIB/log/archives'
    arch_name = 'mi_es&fc&ic&fa&nnic5k1i_N&IN_50r_10&30&100i_1e-4u'
    in_arch_path = 'log/tracking_data'
    tracking_root_dir = './C++/MTF_LIB/log/tracking_data'
    gt_root_dir = '../Datasets'
    out_dir = './C++/MTF_LIB/log/success_rates'
    # set this to '0' to use the original ground truth
    opt_gt_ssm = '0'


    mtf_sm = 'iclk'
    mtf_am = None
    mtf_ssm = None

    mtf_am = 'miIN50r10i8b'
    # mtf_am = 'miIN50r10i1i5000s'

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
    iiw = param_ids['init_identity_warp']

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

    use_cle = 0
    use_opt_gt = 0

    if opt_gt_ssm == '0':
        print 'Using standard ground truth'
        use_opt_gt = 0
    else:
        use_opt_gt = 1
        print 'Using optimized ground truth with ssm: ', opt_gt_ssm
        if opt_gt_ssm == '2':
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

    # err_thresholds = range(1, 20)
    err_thresholds = np.linspace(1, 20, 100)
    seq_ids = range(0, len(sequences))

    n_err_thr = err_thresholds.size
    if actor_id == 0:
        n_seq = 98
    else:
        n_seq = len(seq_ids)

    print 'actor: {:s}'.format(actor)
    print 'mtf_sm: {:s}'.format(mtf_sm)
    print 'mtf_am: {:s}'.format(mtf_am)
    print 'mtf_ssm: {:s}'.format(mtf_ssm)
    print 'iiw: {:d}'.format(iiw)
    print 'use_opt_gt: {:d}'.format(use_opt_gt)
    print 'use_cle: {:d}'.format(use_cle)

    if use_opt_gt:
        out_path = '{:s}/sr_{:s}_{:s}_{:s}_{:s}_{:d}_{:s}.txt'.format(
            out_dir, actor, mtf_sm, mtf_am, mtf_ssm, iiw, opt_gt_ssm)
    else:
        out_path = '{:s}/sr_{:s}_{:s}_{:s}_{:s}_{:d}.txt'.format(
            out_dir, actor, mtf_sm, mtf_am, mtf_ssm, iiw)

    # if os.path.isfile(out_path):
    #     s = raw_input('Output file already exists:\n {:s}\n Proceed with overwrite ?'.format(out_path))
    #     if s == 'n' or s == 'N':
    #         sys.exit()

    arch_fid = None
    if use_arch:
        arch_path = '{:s}/{:s}.zip'.format(arch_root_dir, arch_name)
        print 'Reading tracking data from zip archive: {:s}'.format(arch_path)
        arch_fid = zipfile.ZipFile(arch_path, 'r')

    success_rates = np.zeros((n_err_thr, n_seq + 2), dtype=np.float64)
    success_rates[:, 0] = err_thresholds

    if use_opt_gt:
        gt_dir = '{:s}/{:s}/OptGT'.format(gt_root_dir, actor)
    else:
        gt_dir = '{:s}/{:s}'.format(gt_root_dir, actor)
    print 'Reading GT from folder : {:s}'.format(gt_dir)

    for i in xrange(n_err_thr):
        err_thr = err_thresholds[i]
        success_frame_count = 0
        total_frame_count = 0
        for j in xrange(n_seq):
            seq_id = seq_ids[j]
            seq_name = sequences[seq_id]
            # print 'seq_name: {:s}'.format(seq_name)
            if use_arch:
                tracking_res_path = '{:s}/{:s}/{:s}_{:s}_{:s}_{:d}.txt'.format(
                    in_arch_path, seq_name, mtf_sm, mtf_am, mtf_ssm, iiw)
            else:
                if using_pf:
                    tracking_res_path = '{:s}/pf/{:s}.txt'.format(
                        tracking_root_dir, seq_name)
                else:
                    tracking_res_path = '{:s}/{:s}/{:s}_{:s}_{:s}_{:d}.txt'.format(
                        tracking_root_dir, seq_name, mtf_sm, mtf_am, mtf_ssm, iiw)
                if not os.path.isfile(tracking_res_path):
                    print 'tracking result file not found for: {:s} in: \n {:s} '.format(seq_name, tracking_res_path)
                    sys.exit()
            if use_opt_gt:
                gt_path = '{:s}/{:s}_{:s}.txt'.format(gt_dir, seq_name, opt_gt_ssm)
            else:
                gt_path = '{:s}/{:s}.txt'.format(gt_dir, seq_name)

            success_rates[i, j + 1], success_frames, n_frames = SuccessRate(
                tracking_res_path, gt_path, err_thr, use_cle, arch_fid)
            success_frame_count += success_frames
            total_frame_count += n_frames
        success_rates[i, n_seq + 1] = float(success_frame_count) / float(total_frame_count)
        print 'err_thr: {:15.9f} sfc: {:6d} tfc: {:6d} sr: {:15.9f}'.format(
            err_thr, success_frame_count, total_frame_count, success_rates[i, n_seq + 1])
    print 'Saving success rate data to {:s}'.format(out_path)
    np.savetxt(out_path, success_rates, fmt='%15.12f', delimiter='\t')
    if use_arch:
        arch_fid.close()

        # trackerPath = sys.argv[1]
        # GTPath = sys.argv[2]
        # TH = sys.argv[3]
        # SuccessRate(trackerPath, GTPath, TH)
		
