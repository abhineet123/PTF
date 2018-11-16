from distanceGrid import *
import time
import os
from Misc import getParamDict

if __name__ == '__main__':

    db_root_path = 'E:/UofA/Thesis/#Code/Datasets'
    actor = 'Human'

    params_dict = getParamDict()
    param_ids = readDistGridParams()

    sequences = params_dict['sequences']
    grid_types = params_dict['grid_types']
    inc_types = params_dict['inc_types']
    appearance_models = params_dict['appearance_models']
    tracker_types = params_dict['tracker_types']
    filter_types = params_dict['filter_types']
    opt_types = params_dict['opt_types']

    seq_id = param_ids['seq_id']
    grid_id = param_ids['grid_id']
    appearance_id = param_ids['appearance_id']
    inc_id = param_ids['inc_id']
    tracker_id = param_ids['tracker_id']
    start_id = param_ids['start_id']
    filter_id = param_ids['filter_id']
    kernel_size = param_ids['kernel_size']
    opt_id = param_ids['opt_id']

    n_bins = param_ids['n_bins']
    dof = param_ids['dof']

    arg_id = 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        filter_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        inc_type = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        write_gt_data = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        write_track_data = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        read_dist_data = int(sys.argv[arg_id])
        arg_id += 1

    if seq_id >= len(sequences):
        print 'Invalid dataset_id: ', seq_id
        sys.exit()
    if filter_id >= len(filter_types):
        print 'Invalid filter_id: ', filter_id
        sys.exit()
    if opt_id >= len(opt_types):
        print 'Invalid opt_id: ', opt_id
        sys.exit()

    seq_name = sequences[seq_id]
    filter_type = filter_types[filter_id]
    opt_type = opt_types[opt_id]
    inc_type = inc_types[inc_id]
    appearance_model = appearance_models[appearance_id]

    track_data_folder = 'Tracking Data/' + seq_name + '_' + appearance_model
    dist_folder = 'Distance Data/#PW/' + seq_name + '_' + appearance_model

    if filter_type != 'none':
        track_data_folder = track_data_folder + '_' + filter_type + str(kernel_size)
        dist_folder = dist_folder + '_' + filter_type + str(kernel_size)

    dist_template = inc_type + '_' + opt_type + str(dof)
    new_dist_template = 'pw_'+ opt_type + str(dof)+ '_' + inc_type

    if not os.path.exists(dist_folder):
        raise IOError('Folder containing distance data does not exist: ' + dist_folder)

    if not os.path.exists(track_data_folder):
        raise IOError('Folder containing tracking data does not exist: ' + track_data_folder)

    tracker_type = 'pw_{:s}'.format(dist_template)

    opt_corners_fname = '{:s}/{:s}_corners.txt'.format(track_data_folder, tracker_type)
    opt_params_fname = '{:s}/{:s}_params_inv.txt'.format(track_data_folder, tracker_type)
    errors_fname = '{:s}/{:s}_errors.txt'.format(track_data_folder, tracker_type)

    new_opt_corners_fname = '{:s}/{:s}_corners.txt'.format(track_data_folder, new_dist_template)
    new_opt_params_fname = '{:s}/{:s}_params_inv.txt'.format(track_data_folder, new_dist_template)
    new_errors_fname = '{:s}/{:s}_errors.txt'.format(track_data_folder, new_dist_template)

    trans_dist_fname = dist_folder + '/' + dist_template + '_trans.bin'
    rs_dist_fname = dist_folder + '/' + dist_template + '_rs.bin'
    shear_dist_fname = dist_folder + '/' + dist_template + '_shear.bin'
    proj_dist_fname = dist_folder + '/' + dist_template + '_proj.bin'

    new_trans_dist_fname = dist_folder + '/' + new_dist_template + '_trans.bin'
    new_rs_dist_fname = dist_folder + '/' + new_dist_template + '_rs.bin'
    new_shear_dist_fname = dist_folder + '/' + new_dist_template + '_shear.bin'
    new_proj_dist_fname = dist_folder + '/' + new_dist_template + '_proj.bin'

    os.rename(trans_dist_fname, new_trans_dist_fname)
    os.rename(rs_dist_fname, new_rs_dist_fname)
    os.rename(shear_dist_fname, new_shear_dist_fname)
    os.rename(proj_dist_fname, new_proj_dist_fname)

    os.rename(opt_corners_fname, new_opt_corners_fname)
    os.rename(opt_params_fname, new_opt_params_fname)
    os.rename(errors_fname, new_errors_fname)

