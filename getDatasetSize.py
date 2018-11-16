from Misc import readTrackingData
from Misc import getParamDict
from Misc import readDistGridParams
from Misc import drawRegion

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
    actors = params_dict['actors']
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d.jpg'
    opt_gt_ssm = '0'
    use_opt_gt = 0

    actor_id = 3

    actor = actors[actor_id]
    print 'actor: ', actor

    sequences = sequences[actor]
    n_seq = len(sequences)


    start_id = 0
    end_id = n_seq-1
    print 'n_seq: ', n_seq

    seq_ids = range(start_id, end_id+1)

    out_fname=db_root_dir + '/' + actor + '/'+'n_frames.txt'
    out_file=open(out_fname, 'w')


    total_files=0
    for curr_seq_id in seq_ids:
        seq_name = sequences[curr_seq_id]

        print 'sequence {:d} : {:s}'.format(curr_seq_id, seq_name)

        src_dir = db_root_dir + '/' + actor + '/' + seq_name

        file_list = [each for each in os.listdir(src_dir) if each.endswith('.jpg')]
        n_files = len(file_list)

        print 'n_files: ', n_files
        out_file.write('{:s}\t{:4d}\n'.format(seq_name, n_files))
        total_files += n_files

    out_file.close()
    print 'total_files: {:d}'.format(total_files)













