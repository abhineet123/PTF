import shutil
from Misc import getParamDict
from distanceGrid import *
import os

if __name__ == '__main__':
    params_dict = getParamDict()
    sequences = params_dict['sequences']
    appearance_models = params_dict['appearance_models']
    tracker_types = params_dict['tracker_types']
    filter_types = params_dict['filter_types']

    db_root_path = '../Datasets'
    code_root_path = 'E:/UofA/Thesis/#Code/Tracking_Framework'

    actor = 'Human'
    dist_prarams = readDistGridParams()
    seq_id = dist_prarams['seq_id']
    grid_id = dist_prarams['grid_id']
    appearance_id = dist_prarams['appearance_id']
    inc_id = dist_prarams['inc_id']
    filter_id = dist_prarams['filter_id']
    kernel_size = dist_prarams['kernel_size']
    n_bins = dist_prarams['n_bins']

    start_id = 1
    tracker_id = 0

    arg_id = 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        appearance_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        filter_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        kernel_size = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        tracker_id = int(sys.argv[arg_id])
        arg_id += 1


    if seq_id >= len(sequences):
        print 'Invalid dataset_id: ', seq_id
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
    filter_type = filter_types[filter_id]
    tracker_type = tracker_types[tracker_id]
    appearance_model = appearance_models[appearance_id]

    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    print 'appearance_model: ', appearance_model
    print 'tracker_type: ', tracker_type
    print 'filter_type: ', filter_type
    print 'kernel_size: ', kernel_size

    tracking_data_dir = 'Tracking Data/{:s}_{:s}'.format(appearance_model, seq_name)
    if filter_type != 'none':
        tracking_data_dir = '{:s}_{:s}{:d}'.format(tracking_data_dir, filter_type, kernel_size)

    if not os.path.exists(tracking_data_dir):
        os.mkdir(tracking_data_dir)

    ground_truth_fname = db_root_path + '/' + actor + '/' + seq_name + '.txt'
    new_corners_fname = '{:s}/{:s}_corners.txt'.format(tracking_data_dir, tracker_type)
    if not os.path.isfile(new_corners_fname):
        if tracker_type == 'gt' and os.path.isfile(ground_truth_fname):
            shutil.copy(ground_truth_fname, new_corners_fname)
        else:
            print 'Error: Source corners file not found: ' + ground_truth_fname
            sys.exit()
    else:
        print 'Destination corners file already exists: ', new_corners_fname

    # if tracker_type != 'gt':
    #     in_fname = '{:s}/{:s}_params.txt'.format(tracking_data_dir, tracker_type)
    #     if not os.path.isfile(in_fname):
    #         print 'Warning: New tracking params file is not present: ' + in_fname
    #         old_fname = '{:s}/{:s}_params.txt'.format(tracking_data_dir, seq_name)
    #         if not os.path.isfile(old_fname):
    #             print 'Error: Old tracking params file is not present: ' + old_fname
    #             sys.exit()
    #         print 'Renaming ' + old_fname
    #         os.rename(old_fname, in_fname)
    #     out_fname = '{:s}/{:s}_params_inv.txt'.format(tracking_data_dir, tracker_type)
    #     dec_params_mat = reformatHomFile(in_fname, out_fname)

    ground_truth = readTrackingData(new_corners_fname)
    no_of_frames = ground_truth.shape[0]
    print 'no_of_frames: ', no_of_frames
    end_id = no_of_frames

    prev_corners = np.asarray([ground_truth[start_id - 1, 0:2].tolist(),
                               ground_truth[start_id - 1, 2:4].tolist(),
                               ground_truth[start_id - 1, 4:6].tolist(),
                               ground_truth[start_id - 1, 6:8].tolist()]).T
    (prev_corners_norm, prev_norm_mat) = getNormalizedPoints(prev_corners)

    params_inv = []
    params_fwd = []

    for frame_id in xrange(start_id, end_id):
        print 'frame_id: ', frame_id
        curr_corners = np.asarray([ground_truth[frame_id, 0:2].tolist(),
                                   ground_truth[frame_id, 2:4].tolist(),
                                   ground_truth[frame_id, 4:6].tolist(),
                                   ground_truth[frame_id, 6:8].tolist()]).T

        curr_corners_norm, curr_norm_mat = getNormalizedPoints(curr_corners)
        try:
            curr_hom_mat = np.mat(util.compute_homography(prev_corners_norm, curr_corners_norm))
        except np.linalg.linalg.LinAlgError as error_msg:
            print'Error encountered while computing homography for frame {:d}: {:s}'.format(frame_id, error_msg)
            break
        tx_inv, ty_inv, theta_inv, scale_inv, a_inv, b_inv, v1_inv, v2_inv = getHomographyParamsInverse(curr_hom_mat)
        params_inv.append([frame_id + 1, tx_inv, ty_inv, theta_inv, scale_inv, a_inv, b_inv, v1_inv, v2_inv])

        tx_fwd, ty_fwd, theta_fwd, scale_fwd, a_fwd, b_fwd, v1_fwd, v2_fwd = getHomographyParamsForward(curr_hom_mat)
        params_fwd.append([frame_id + 1, tx_fwd, ty_fwd, theta_fwd, scale_fwd, a_fwd, b_fwd, v1_fwd, v2_fwd])

        prev_corners = curr_corners.copy()
        prev_corners_norm = curr_corners_norm.copy()

    params_inv_fname = '{:s}/{:s}_params_inv.txt'.format(tracking_data_dir, tracker_type)
    params_fwd_fname = '{:s}/{:s}_params_fwd.txt'.format(tracking_data_dir, tracker_type)
    np.savetxt(params_inv_fname, np.array(params_inv), fmt='%12.8f', delimiter='\t')
    np.savetxt(params_fwd_fname, np.array(params_fwd), fmt='%12.8f', delimiter='\t')












