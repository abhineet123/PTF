from distanceGrid import *
import time
import os
from Misc import *
import shutil

if __name__ == '__main__':

    db_root_dir = '../Datasets'
    track_root_dir = '../Tracking Data'
    img_root_dir = '../Image Data'
    dist_root_dir = '../Distance Data'

    params_dict = getParamDict()
    param_ids = readDistGridParams()

    actors = params_dict['actors']
    sequences = params_dict['sequences']
    grid_types = params_dict['grid_types']
    inc_types = params_dict['inc_types']
    appearance_models = params_dict['appearance_models']
    tracker_types = params_dict['tracker_types']
    filter_types = params_dict['filter_types']
    challenges = params_dict['challenges']

    actor_id = param_ids['actor_id']
    seq_id = param_ids['seq_id']
    grid_id = param_ids['grid_id']
    appearance_id = param_ids['appearance_id']
    inc_id = param_ids['inc_id']
    tracker_id = param_ids['tracker_id']
    start_id = param_ids['start_id']
    filter_id = param_ids['filter_id']
    kernel_size = param_ids['kernel_size']
    n_bins = param_ids['n_bins']
    challenge_id = param_ids['challenge_id']


    std_resx = 50
    std_resy = 50
    n_pts = std_resx * std_resy

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
        tracker_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        filter_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        inc_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        actor_id = int(sys.argv[arg_id])
        arg_id += 1

    if actor_id >= len(actors):
        print 'Invalid actor_id: ', actor_id
        sys.exit()

    actor = actors[actor_id]
    sequences = sequences[actor]

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
    # print 'inc_types: ', inc_types
    inc_type = inc_types[inc_id]
    challenge = challenges[challenge_id]

    if actor == 'METAIO':
        seq_name = seq_name + '_' + challenge

    print 'actor: ', actor
    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    print 'grid_type: ', grid_type
    print 'appearance_model: ', appearance_model
    print 'inc_type: ', inc_type
    print 'tracker_type: ', tracker_type
    print 'filter_type: ', filter_type
    print 'kernel_size: ', kernel_size
    print 'n_bins: ', n_bins

    src_folder = db_root_dir + '/' + actor + '/' + seq_name

    ground_truth_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'

    ground_truth = readTrackingData(ground_truth_fname)
    no_of_frames = ground_truth.shape[0]
    print 'no_of_frames: ', no_of_frames

    printScalarToFile(seq_name, 'seq_name', './C++/log/py_log.txt', fmt='{:s}', mode='w')

    end_id = no_of_frames

    init_corners = np.asarray([ground_truth[0, 0:2].tolist(),
                               ground_truth[0, 2:4].tolist(),
                               ground_truth[0, 4:6].tolist(),
                               ground_truth[0, 6:8].tolist()]).T

    std_pts, std_corners = getNormalizedUnitSquarePts(std_resx, std_resy, 0.5)
    std_corners_hm = util.homogenize(std_corners)

    (init_corners_norm, init_norm_mat_inv) = getNormalizedPoints(init_corners)
    norm_mean_dist = np.mean(np.sqrt(np.sum(np.square(init_corners_norm), axis=1)))
    print 'norm_mean_dist:', norm_mean_dist
    init_corners_hm = np.mat(util.homogenize(init_corners_norm))
    init_mean_dist = np.mean(np.sqrt(np.sum(np.square(init_corners_norm), axis=0)))

    affine_mat = computeAffineLS(std_corners, init_corners_norm)

    rec_corners = util.dehomogenize(affine_mat * std_corners_hm)

    printMatrixToFile(init_corners, 'init_corners', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    printMatrixToFile(std_corners, 'std_corners', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    printMatrixToFile(init_corners_norm, 'init_corners_norm', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    printMatrixToFile(rec_corners, 'rec_corners', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    printMatrixToFile(init_norm_mat_inv, 'init_norm_mat_inv', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')

