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
    n_seq = 100

    for seq_id in xrange(n_seq):
        seq_name = sequences[seq_id]
        # seq_name = 'nl_mugII_s1'

        print 'seq_name: ', seq_name

        src_fname = db_root_dir + '/' + actor + '/' + seq_name + '/groundtruth_rect.txt'
        dst_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'

        if not os.path.isfile(src_fname):
            print 'The source ground truth file : {:s} does not exist'.format(src_fname)
            continue
        data_file = open(src_fname, 'r')
        lines = data_file.readlines()
        data_file.close()

        print 'Ground truth file has {:d} lines'.format(len(lines))

        dst_file = open(dst_fname, 'w')
        dst_file.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
        line_id = 1
        for line in lines:
            words = line.split(',')
            if len(words) != 4:
                words = line.split()
                if len(words) != 4:
                    raise StandardError('invalid formatting on line {:d} : {:s}'.format(line_id, line))
            x = float(words[0])
            y = float(words[1])
            width = float(words[2])
            height = float(words[3])

            lx = x
            rx = x + width
            uy = y
            ly = y + height

            # lx = x - width / 2.0
            # rx = x + width / 2.0
            # uy = y - height / 2.0
            # ly = y + height / 2.0

            corner_str='{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}'.format(
                lx, uy, rx, uy, rx, ly, lx, ly)
            dst_file.write('frame{:05d}.jpg\t{:s}\n'.format(line_id, corner_str))
            line_id+=1
        dst_file.close()



