import sys
import os
import numpy as np
from Misc import *

if __name__ == '__main__':

    appearance_models = {0: 'ssd',
                         1: 'mi',
                         2: 'ncc',
                         3: 'scv',
                         4: 'ccre',
                         5: 'mi2',
                         6: 'ncc2',
                         7: 'scv2',
                         8: 'mi_new'
    }
    tracker_types = {0: 'gt',
                     1: 'esm',
                     2: 'ic',
                     3: 'nnic',
                     4: 'pf'
    }
    grid_types = {0: 'trans',
                  1: 'rs',
                  2: 'shear',
                  3: 'proj',
                  4: 'rtx',
                  5: 'rty',
                  6: 'stx',
                  7: 'sty'
    }
    filter_types = {0: 'none',
                    1: 'gauss',
                    2: 'box',
                    3: 'norm_box',
                    4: 'bilateral',
                    5: 'median',
                    6: 'gabor',
                    7: 'sobel',
                    8: 'scharr',
                    9: 'LoG',
                    10: 'DoG',
                    11: 'laplacian',
                    12: 'canny'
    }
    sequences = {0: 'nl_bookI_s3',
                 1: 'nl_bookII_s3',
                 2: 'nl_bookIII_s3',
                 3: 'nl_cereal_s3',
                 4: 'nl_juice_s3',
                 5: 'nl_mugI_s3',
                 6: 'nl_mugII_s3',
                 7: 'nl_mugIII_s3',

                 8: 'nl_bookI_s4',
                 9: 'nl_bookII_s4',
                 10: 'nl_bookIII_s4',
                 11: 'nl_cereal_s4',
                 12: 'nl_juice_s4',
                 13: 'nl_mugI_s4',
                 14: 'nl_mugII_s4',
                 15: 'nl_mugIII_s4',

                 16: 'nl_bus',
                 17: 'nl_highlighting',
                 18: 'nl_letter',
                 19: 'nl_newspaper',
    }

    db_root_path = 'E:/UofA/Thesis/#Code/Datasets'
    actor = 'Human'
    seq_id = 3
    inc_type = 'ic'
    grid_id = 0
    appearance_id = 0
    tracker_id = 0
    start_id = 1
    filter_id = 0
    kernel_size = 9

    arg_id = 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        grid_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        appearance_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        filter_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        tracker_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        inc_type = sys.argv[arg_id]
        arg_id += 1

    if seq_id >= len(sequences):
        print 'Invalid dataset_id: ', seq_id
        sys.exit()
    if grid_id >= len(grid_types):
        print 'Invalid grid_id: ', grid_id
        sys.exit()
    if appearance_id >= len(appearance_models):
        print 'Invalid appearance_id: ', appearance_id
        sys.exit()
    if filter_id >= len(filter_types):
        print 'Invalid filter_id: ', filter_id
        sys.exit()
    if tracker_id >= len(tracker_types):
        print 'Invalid tracker_id: ', tracker_id
        sys.exit()

    seq_name = sequences[seq_id]
    grid_type = grid_types[grid_id]
    filter_type = filter_types[filter_id]
    tracker_type = tracker_types[tracker_id]
    appearance_model = appearance_models[appearance_id]

    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    print 'inc_type: ', inc_type
    print 'grid_type: ', grid_type
    print 'filter_type: ', filter_type
    print 'kernel_size: ', kernel_size
    print 'tracker_type: ', tracker_type
    print 'appearance_model: ', appearance_model

    src_folder = db_root_path + '/' + actor + '/' + seq_name

    if tracker_type != 'gt':
        ground_truth_fname = db_root_path + '/' + actor + '/' + seq_name + '_' + tracker_type + '.txt'
    else:
        ground_truth_fname = db_root_path + '/' + actor + '/' + seq_name + '.txt'

    ground_truth = readTrackingData(ground_truth_fname)
    no_of_frames = ground_truth.shape[0]
    print 'no_of_frames: ', no_of_frames
    end_id = no_of_frames

    img_folder = 'Image Data'
    if filter_type != 'none':
        img_fname = img_folder + '/' + seq_name + '_' + filter_type + str(kernel_size) + '.bin'
        root_dist_folder = 'Distance Data/' + seq_name + '_' + appearance_model + '_' + tracker_type + '_' + filter_type + str(
            kernel_size)
    else:
        img_fname = img_folder + '/' + seq_name + '.bin'
        root_dist_folder = 'Distance Data/' + seq_name + '_' + appearance_model + '_' + tracker_type

    dist_template = inc_type + '_' + grid_type

    src_dist_folder = root_dist_folder + '/' + dist_template
    if not os.path.exists(src_dist_folder):
        raise IOError('The source distance folder does not exist')

    dist_fname = root_dist_folder + '/' + dist_template + '.bin'
    if os.path.isfile(dist_fname):
        s = raw_input('\nWarning: The distance file already exists. Proceed with overwrite ?\n')
        if s == 'n' or s == 'N':
            sys.exit()
    dist_fid = open(dist_fname, 'wb')

    if grid_type == 'trans':
        y_vec = np.loadtxt(src_dist_folder + '/tx_vec.txt')
        x_vec = np.loadtxt(src_dist_folder + '/ty_vec.txt')
        print 'tx_vec: ', y_vec
        print 'ty_vec: ', x_vec
    elif grid_type == 'rtx':
        y_vec = np.loadtxt(src_dist_folder + '/tx_vec.txt')
        x_vec = np.loadtxt(src_dist_folder + '/theta_vec.txt')
        print 'tx_vec: ', y_vec
        print 'theta_vec: ', x_vec
    elif grid_type == 'rty':
        y_vec = np.loadtxt(src_dist_folder + '/ty_vec.txt')
        x_vec = np.loadtxt(src_dist_folder + '/theta_vec.txt')
        print 'ty_vec: ', y_vec
        print 'theta_vec: ', x_vec
    elif grid_type == 'rs':
        y_vec = np.loadtxt(src_dist_folder + '/scale_vec.txt')
        x_vec = np.loadtxt(src_dist_folder + '/theta_vec.txt')
        print 'scale_vec: ', y_vec
        print 'theta_vec: ', x_vec
    elif grid_type == 'shear':
        y_vec = np.loadtxt(src_dist_folder + '/a_vec.txt')
        x_vec = np.loadtxt(src_dist_folder + '/b_vec.txt')
        print 'a_vec: ', y_vec
        print 'b_vec: ', x_vec
    elif grid_type == 'proj':
        y_vec = np.loadtxt(src_dist_folder + '/v1_vec.txt')
        x_vec = np.loadtxt(src_dist_folder + '/v2_vec.txt')
        print 'v1_vec: ', y_vec
        print 'v2_vec: ', x_vec
    else:
        raise StandardError('Invalid grid_type: ' + grid_type)

    np.array([start_id - 1, start_id, x_vec.size, y_vec.size], dtype=np.uint32).tofile(dist_fid)
    x_vec.tofile(dist_fid)
    y_vec.tofile(dist_fid)

    dist_grid_size = x_vec.size * y_vec.size

    for frame_id in xrange(start_id, end_id):
        try:
            src_dist_fid = open(src_dist_folder + '/' + 'dist_grid_' + str(frame_id) + '.bin', 'rb')
        except IOError:
            print 'Source distance file does not exist for frame_id: ', frame_id
            break
        dist_grid = np.fromfile(src_dist_fid, dtype=np.float64, count=dist_grid_size)
        src_dist_fid.close()

        dist_grid.tofile(dist_fid)
        current_offset = dist_fid.tell()
        dist_fid.seek(0)
        np.array([frame_id], dtype=np.uint32).tofile(dist_fid)
        dist_fid.seek(current_offset)
        print 'frame_id:\t{:-5d}'.format(frame_id)

    dist_fid.close()
