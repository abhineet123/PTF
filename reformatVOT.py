from Misc import getParamDict
from Misc import readDistGridParams

import sys
import cv2
import numpy as np
import os

if __name__ == '__main__':

    params_dict = getParamDict()
    param_ids = readDistGridParams()
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'

    seq_id = param_ids['seq_id']
    actor = 'VOT16'
    sequences = sequences[actor]
    n_seq = 60

    for seq_id in xrange(n_seq):
        seq_name = sequences[seq_id]
        # seq_name = 'nl_mugII_s1'

        print 'seq_name: ', seq_name

        src_fname = db_root_dir + '/' + actor + '/' + seq_name + '/groundtruth.txt'
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
            if len(words) != 8:
                raise StandardError('invalid formatting on line {:d} : {:s}'.format(line_id, line))

            x_coords = np.array([float(words[0]), float(words[2]), float(words[4]), float(words[6])])
            y_coords = np.array([float(words[1]), float(words[3]), float(words[5]), float(words[7])])
            y_sorted_idx = np.argsort(y_coords)

            if x_coords[y_sorted_idx[0]] < x_coords[y_sorted_idx[1]]:
                ulx = x_coords[y_sorted_idx[0]]
                uly = y_coords[y_sorted_idx[0]]
                urx = x_coords[y_sorted_idx[1]]
                ury = y_coords[y_sorted_idx[1]]
            else:
                ulx = x_coords[y_sorted_idx[1]]
                uly = y_coords[y_sorted_idx[1]]
                urx = x_coords[y_sorted_idx[0]]
                ury = y_coords[y_sorted_idx[0]]

            if x_coords[y_sorted_idx[2]] < x_coords[y_sorted_idx[3]]:
                llx = x_coords[y_sorted_idx[2]]
                lly = y_coords[y_sorted_idx[2]]
                lrx = x_coords[y_sorted_idx[3]]
                lry = y_coords[y_sorted_idx[3]]
            else:
                llx = x_coords[y_sorted_idx[3]]
                lly = y_coords[y_sorted_idx[3]]
                lrx = x_coords[y_sorted_idx[2]]
                lry = y_coords[y_sorted_idx[2]]

            # llx = float(words[0])
            # lly = float(words[1])
            # ulx = float(words[2])
            # uly = float(words[3])
            # urx = float(words[4])
            # ury = float(words[5])
            # lrx = float(words[6])
            # lry = float(words[7])

            corner_str = '{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}'.format(
                ulx, uly, urx, ury, lrx, lry, llx, lly)
            dst_file.write('frame{:05d}.jpg\t{:s}\n'.format(line_id, corner_str))
            line_id += 1
        dst_file.close()



