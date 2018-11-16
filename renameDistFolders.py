from distanceGrid import *
import time
import os
from Misc import getParamDict

if __name__ == '__main__':
    params_dict = getParamDict()
    sequences = params_dict['sequences']
    appearance_models = params_dict['appearance_models']
    filter_types = params_dict['filter_types']

    db_root_path = 'E:/UofA/Thesis/#Code/Datasets'
    actor = 'Human'
    dist_prarams = readDistGridParams()
    seq_id = dist_prarams['seq_id']
    appearance_id = dist_prarams['appearance_id']
    filter_id = dist_prarams['filter_id']
    kernel_size = dist_prarams['kernel_size']

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

    if seq_id >= len(sequences):
        print 'Invalid dataset_id: ', seq_id
        sys.exit()
    if appearance_id >= len(appearance_models):
        print 'Invalid appearance_id: ', appearance_id
        sys.exit()
    if filter_id >= len(filter_types):
        print 'Invalid filter_id: ', filter_id
        sys.exit()

    seq_name = sequences[seq_id]
    appearance_model = appearance_models[appearance_id]
    filter_type = filter_types[filter_id]

    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    print 'appearance_model: ', appearance_model
    print 'filter_type: ', filter_type
    print 'kernel_size: ', kernel_size

    src_folder = db_root_path + '/' + actor + '/' + seq_name

    root_dir = 'Tracking Data'

    dist_folder = root_dir + '/' + seq_name + '_' + appearance_model
    new_dist_folder = root_dir + '/' + appearance_model + '_' + seq_name
    if filter_type != 'none':
        dist_folder = dist_folder + '_' + filter_type + str(kernel_size)
        new_dist_folder = new_dist_folder + '_' + filter_type + str(kernel_size)

    if not os.path.exists(dist_folder):
        raise IOError('Distance data folder does not exist: ' + dist_folder)

    os.rename(dist_folder, new_dist_folder)
