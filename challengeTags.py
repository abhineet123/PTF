import numpy as np
import math
import sys

from Misc import getParamDict
from Misc import readDistGridParams
import os

'''
USAGE: python <path of tracked file> <path of ground truth> <path of tag files>
     : Notice TH (threshold) is set at 5, you can change it if you want

OUTPUT: Success rate as a fraction of the total frames tracked
'''


def frameTagAnalysis(trackerPath, GTPath, tagPath, TH):
    # trackerPath = sys.argv[1]
    # GTPath = sys.argv[2]
    # tagPath = sys.argv[3]
    # TH = 5
    load_Tracker = open(trackerPath, 'r').readlines()
    load_GT = open(GTPath, 'r').readlines()
    load_tags = open(tagPath, 'r').readlines()

    no_frames = len(load_Tracker) - 1
    E = 0
    I = 1


    # Challenges #
    TR = 0
    RO = 0
    PR = 0
    SR = 0
    OC = 0
    TX = 0
    BL = 0
    SC = 0
    tag_count = np.zeros(8, dtype=np.uint32)

    while I < no_frames:
        Tracker = load_Tracker[I].strip().split()
        GT = load_GT[I].strip().split()
        tags = load_tags[I].strip().split()
        err = 0

        # Alignment error
        # print I
        for p in range(1, 9):
            err = (float(Tracker[p]) - float(GT[p])) ** 2 + err
        err = math.sqrt(err / 4)
        if err < TH:
            if int(float(tags[1])) == 1:
                tag_count[0] += 1
                TR += 1
            if int(float(tags[2])) == 1:
                tag_count[1] += 1
                RO += 1
            if int(float(tags[3])) == 1:
                tag_count[2] += 1
                PR += 1
            if int(float(tags[4])) == 1:
                tag_count[3] += 1
                SR += 1
            if int(float(tags[5])) == 1:
                tag_count[4] += 1
                OC += 1
            if int(float(tags[6])) == 1:
                tag_count[5] += 1
                TX += 1
            if int(float(tags[7])) == 1:
                tag_count[6] += 1
                BL += 1
            if int(float(tags[8])) == 1:
                tag_count[7] += 1
                SC += 1

        I += 1

    # print 'TR:', TR, ' RO:', RO, ' PR:', PR, ' SR:', SR, ' OC:', OC, ' TX:', TX, ' BL:', BL, ' SC:', SC
    return tag_count


if __name__ == '__main__':
    tracking_res_dir = './C++/MTF_LIB/log/tracking_data'
    gt_dir = '../Datasets'
    out_dir = './C++/MTF_LIB/log'

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
        opt_gt_ssm = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        mtf_sm_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        mtf_am_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        mtf_ssm_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        iiw = int(sys.argv[arg_id])
        arg_id += 1

    mtf_sm = mtf_sms[mtf_sm_id]
    mtf_am = mtf_ams[mtf_am_id]
    mtf_ssm = mtf_ssms[mtf_ssm_id]

    # err_thresholds = range(1, 20)
    err_thr = 5
    print 'err_thr: {:15.9f}'.format(err_thr)
    actor = actors[actor_id]
    sequences = sequences[actor]

    seq_ids = range(0, len(sequences))
    n_seq = len(seq_ids)

    print 'actor: {:s}'.format(actor)
    print 'mtf_sm: {:s}'.format(mtf_sm)
    print 'mtf_am: {:s}'.format(mtf_am)
    print 'mtf_ssm: {:s}'.format(mtf_ssm)
    print 'iiw: {:d}'.format(iiw)

    n_tags = 8

    seq_tags = np.zeros((n_tags, n_seq + 1), dtype=np.uint32)
    for j in xrange(n_seq):
        seq_id = seq_ids[j]
        seq_name = sequences[seq_id]
        print '{:s}'.format(seq_name)
        tracking_res_path = '{:s}/{:s}/{:s}_{:s}_{:s}_{:d}.txt'.format(
            tracking_res_dir, seq_name, mtf_sm, mtf_am, mtf_ssm, iiw)
        gt_path = '{:s}/{:s}/{:s}.txt'.format(gt_dir, actor, seq_name)
        tag_path='{:s}/{:s}/tags/{:s}.txt'.format(gt_dir, actor, seq_name)
        if not os.path.isfile(tracking_res_path):
            print 'tracking result file not found for: {:s}'.format(seq_name)
            sys.exit(0)
        else:
            seq_tags[:, j + 1] = frameTagAnalysis(tracking_res_path, gt_path, tag_path, err_thr)
    np.savetxt('{:s}/tags_{:s}_{:s}_{:s}_{:d}.txt'.format(out_dir, mtf_sm, mtf_am, mtf_ssm, iiw), seq_tags, fmt='%6d', delimiter='\t')
