from Misc import getParamDict
from utility import dehomogenize
from utility import homogenize
import numpy as np
import os

def arrangeCorners(words):
    x_coords = np.array([float(words[0]), float(words[2]), float(words[4]), float(words[6])])
    y_coords = np.array([float(words[1]), float(words[3]), float(words[5]), float(words[7])])
    y_sorted_idx = np.argsort(y_coords)

    idx = range(4)

    if x_coords[y_sorted_idx[0]] < x_coords[y_sorted_idx[1]]:
        ulx = x_coords[y_sorted_idx[0]]
        uly = y_coords[y_sorted_idx[0]]
        urx = x_coords[y_sorted_idx[1]]
        ury = y_coords[y_sorted_idx[1]]
        idx[0] = y_sorted_idx[0]
        idx[1] = y_sorted_idx[1]
    else:
        ulx = x_coords[y_sorted_idx[1]]
        uly = y_coords[y_sorted_idx[1]]
        urx = x_coords[y_sorted_idx[0]]
        ury = y_coords[y_sorted_idx[0]]
        idx[0] = y_sorted_idx[1]
        idx[1] = y_sorted_idx[0]

    if x_coords[y_sorted_idx[2]] < x_coords[y_sorted_idx[3]]:
        llx = x_coords[y_sorted_idx[2]]
        lly = y_coords[y_sorted_idx[2]]
        lrx = x_coords[y_sorted_idx[3]]
        lry = y_coords[y_sorted_idx[3]]
        idx[2] = y_sorted_idx[2]
        idx[3] = y_sorted_idx[3]
    else:
        llx = x_coords[y_sorted_idx[3]]
        lly = y_coords[y_sorted_idx[3]]
        lrx = x_coords[y_sorted_idx[2]]
        lry = y_coords[y_sorted_idx[2]]
        idx[2] = y_sorted_idx[3]
        idx[3] = y_sorted_idx[2]

    corners = [
        [ulx, urx, lrx, llx],
        [uly, ury, lry, lly]
        ]

    return corners, idx

if __name__ == '__main__':

    params_dict = getParamDict()
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'

    actor = 'PTW'
    sequences = sequences[actor]
    start_id = 49
    end_id = 62

    for seq_id in xrange(start_id, end_id + 1):
        seq_name = sequences[seq_id]
        # seq_name = 'nl_mugII_s1'

        print 'seq_name: ', seq_name

        gt_hom_fname = db_root_dir + '/' + actor + '/annotation/' + seq_name + '_gt_homography.txt'
        gt_pts_fname = db_root_dir + '/' + actor + '/annotation/' + seq_name + '_gt_points.txt'
        dst_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'

        if not os.path.isfile(gt_hom_fname):
            print 'The source ground truth file : {:s} does not exist'.format(gt_hom_fname)
            continue
        if not os.path.isfile(gt_pts_fname):
            print 'The source ground truth file : {:s} does not exist'.format(gt_pts_fname)
            continue

        gt_hom_file = open(gt_hom_fname, 'r')
        gt_hom_lines = gt_hom_file.readlines()
        gt_hom_file.close()

        gt_pts_file = open(gt_pts_fname, 'r')
        gt_pts_lines = gt_pts_file.readlines()
        gt_pts_file.close()

        words = gt_pts_lines[0].rstrip().split()
        if len(words) != 8:
                raise StandardError('invalid formatting on first line of line of points gt file : {:s}'.format(init_location))
        corners, idx = arrangeCorners(words)
        init_location = np.matrix(idx)
        init_location_hom = homogenize(init_location)

        # print 'init_location: ', init_location
        # print 'init_location_hom: ', init_location_hom

        print 'Ground truth file has {:d} lines'.format(len(gt_hom_lines))

        dst_file = open(dst_fname, 'w')
        dst_file.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
        curr_location_list =[]

        line_id = 0
        for line in gt_pts_lines:
            words = line.rstrip().split()
            # if len(words) != 9:
            #     raise StandardError('invalid formatting on line {:d} of homography gt file : {:s}'.format(
            # line_id, line))
            # hom_mat = np.matrix([
            #     [float(words[0]),  float(words[1]),  float(words[2])],
            #     [float(words[3]),  float(words[4]),  float(words[5])],
            #     [float(words[6]),  float(words[7]),  float(words[8])]
            # ])
            # curr_location = dehomogenize(hom_mat * init_location_hom)
            # print 'hom_mat: ', hom_mat

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
        for frame_id in xrange(n_frames):
            if np.isnan(curr_location_list[frame_id]).any():
                curr_location = np.matrix([
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ])
                # if frame_id == n_frames - 1:
                #     curr_location = curr_location_list[frame_id - 1]
                # else:
                #     curr_location = (curr_location_list[frame_id - 1] +
                #                      curr_location_list[frame_id + 1])/2.0
            else:
                curr_location = curr_location_list[frame_id]

            corner_str = '{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}'.format(
                curr_location[0, 0], curr_location[1, 0], curr_location[0, 1], curr_location[1, 1],
                curr_location[0, 2], curr_location[1, 2], curr_location[0, 3], curr_location[1, 3])

            # print 'corner_str: ', corner_str
            dst_file.write('frame{:05d}.jpg\t{:s}\n'.format(frame_id + 1, corner_str))

        dst_file.close()



