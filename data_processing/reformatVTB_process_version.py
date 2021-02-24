from Misc import getParamDict
from Misc import readDistGridParams

import sys
import cv2
import numpy as np
import os

if __name__ == '__main__':

    params_dict = getParamDict()
    param_ids = readDistGridParams()
    pause_seq = 0
    gt_col = (0, 0, 255)
    gt_thickness = 1
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d.jpg'
    opt_gt_ssm = '0'
    use_opt_gt = 0

    seq_id = param_ids['seq_id']

    actor = 'VTB'
    sequences = sequences[actor]
    seq_name = sequences[seq_id]
    # seq_name = 'nl_mugII_s1'

    print 'actor: ', actor
    print 'seq_name: ', seq_name

    src_dir = db_root_dir + '/' + actor + '/' + seq_name + '/img'
    dst_dir = db_root_dir + '/' + actor + '/' + seq_name

    if not os.path.exists(src_dir):
        raise StandardError('The source directory : {:s} does not exist'.format(src_dir))

    file_list = [each for each in os.listdir(src_dir) if each.endswith('.jpg')]
    file_list.sort()

    n_files = len(file_list)

    print 'renaming {:d} files...'.format(n_files)
    print file_list

    file_id = 1
    for fname in file_list:
        old_fname = '{:s}/{:s}'.format(src_dir, fname)
        new_fname = '{:s}/frame{:05d}.jpg'.format(dst_dir, file_id)
        file_id += 1
        os.rename(old_fname, new_fname)