from distanceGrid import *
import time
import os
from Misc import getParamDict

if __name__ == '__main__':
    params_dict = getParamDict()
    sequences = params_dict['sequences']
    grid_types = params_dict['grid_types']
    inc_types = params_dict['inc_types']
    appearance_models = params_dict['appearance_models']
    tracker_types = params_dict['tracker_types']
    filter_types = params_dict['filter_types']

    db_root_path = 'E:/UofA/Thesis/#Code/Datasets'
    actor = 'Human'
    dist_prarams = readDistGridParams()
    seq_id = dist_prarams['seq_id']
    grid_id = dist_prarams['grid_id']
    appearance_id = dist_prarams['appearance_id']
    inc_id = dist_prarams['inc_id']
    tracker_id = dist_prarams['tracker_id']
    start_id = dist_prarams['start_id']
    filter_id = dist_prarams['filter_id']
    kernel_size = dist_prarams['kernel_size']
    n_bins = dist_prarams['n_bins']
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
        inc_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        tracker_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        filter_id = int(sys.argv[arg_id])
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
    inc_type = inc_types[inc_id]

    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    print 'grid_type: ', grid_type
    print 'appearance_model: ', appearance_model
    print 'inc_type: ', inc_type
    print 'tracker_type: ', tracker_type
    print 'filter_type: ', filter_type
    print 'kernel_size: ', kernel_size

    src_folder = db_root_path + '/' + actor + '/' + seq_name

    root_dir = 'Distance Data'

    dist_folder = root_dir + '/' + seq_name + '_' + appearance_model + '_' + tracker_type
    new_dist_folder = root_dir + '/' + + seq_name + '_' + appearance_model
    if filter_type != 'none':
        dist_folder = dist_folder + '_' + filter_type + str(kernel_size)
        new_dist_folder = new_dist_folder + '_' + filter_type + str(kernel_size)

    if not os.path.exists(dist_folder):
        raise IOError('Folder containing distance data does not exist: ' + dist_folder)

    if not os.path.exists(new_dist_folder):
        os.makedirs(new_dist_folder)

    dist_fname = dist_folder + '/' + inc_type + '_' + grid_type + '.bin'
    new_dist_fname = new_dist_folder + '/' + tracker_type + '_' + inc_type + '_' + grid_type + '.bin'

    os.rename(dist_fname, new_dist_fname)
