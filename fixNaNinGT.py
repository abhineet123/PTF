from Misc import getParamDict
import numpy as np
import os
import shutil

if __name__ == '__main__':

    params_dict = getParamDict()
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'

    actor = 'PTW'
    sequences = sequences[actor]
    start_id = 0
    end_id = 209
    overwrite_gt = 1

    for seq_id in xrange(start_id, end_id + 1):
        seq_name = sequences[seq_id]
        # seq_name = 'nl_mugII_s1'

        # print 'seq_name: ', seq_name

        gt_corners_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
        gt_back_fname = db_root_dir + '/' + actor + '/' + seq_name + '.back_nan'

        if not os.path.isfile(gt_corners_fname):
            print 'The source ground truth file : {:s} does not exist'.format(gt_corners_fname)
            continue

        gt_pts_file = open(gt_corners_fname, 'r')
        gt_pts_lines = gt_pts_file.readlines()
        gt_pts_file.close()

        del gt_pts_lines[0]

        # print 'Ground truth file has {:d} lines'.format(len(gt_pts_lines))
        if overwrite_gt:
            corrected_corners_fname = gt_corners_fname
        else:
            corrected_corners_fname = db_root_dir + '/' + actor + '/' + seq_name + '_corr.txt'

        if overwrite_gt and os.path.isfile(gt_corners_fname):
            backup_gt_fname = gt_corners_fname.replace('.txt', '.back_ptf')
            print 'Backing up existing GT to {:s}'.format(backup_gt_fname)
            shutil.move(gt_corners_fname, backup_gt_fname)

        dst_file = open(corrected_corners_fname, 'w')
        dst_file.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
        curr_location_list =[]

        line_id = 0
        for line in gt_pts_lines:
            words = line.rstrip().split()
            del words[0]

            if len(words) != 8:
                raise StandardError('invalid formatting on line {:d} of points gt file : {:s}'.format(
            line_id, line))

            curr_location = np.matrix([
                [float(words[0]), float(words[2]), float(words[4]), float(words[6])],
                [float(words[1]), float(words[3]), float(words[5]), float(words[7])]
            ])
            # print 'curr_location: ', curr_location
            curr_location_list.append(curr_location)
            line_id += 1


        n_frames = len(curr_location_list)
        n_nans_in_gt = 0
        for frame_id in xrange(n_frames):
            if np.isnan(curr_location_list[frame_id]).any():
                n_nans_in_gt += 1
                if frame_id == n_frames - 1:
                    curr_location = curr_location_list[frame_id - 1]
                else:
                    curr_location = (curr_location_list[frame_id - 1] +
                                     curr_location_list[frame_id + 1])/2.0
            else:
                curr_location = curr_location_list[frame_id]

            corner_str = '{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}'.format(
                curr_location[0, 0], curr_location[1, 0], curr_location[0, 1], curr_location[1, 1],
                curr_location[0, 2], curr_location[1, 2], curr_location[0, 3], curr_location[1, 3])

            # print 'corner_str: ', corner_str
            dst_file.write('frame{:05d}.jpg\t{:s}\n'.format(frame_id + 1, corner_str))

        dst_file.close()
        if n_nans_in_gt>0:
            print '{:d} NaNs found in {:s}'.format(n_nans_in_gt,seq_name)



