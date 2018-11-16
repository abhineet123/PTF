from distanceGrid import *
import time
import os
from Misc import getParamDict
import shutil

if __name__ == '__main__':

    db_root_dir = '../Datasets'
    out_root_dir = './ssm_sigma'

    params_dict = getParamDict()

    actors = params_dict['actors']
    sequences = params_dict['sequences']
    inc_types = params_dict['inc_types']
    appearance_models = params_dict['appearance_models']
    challenges = params_dict['challenges']

    actor_id = 0
    seq_id = 10

    start_id = 1

    arg_id = 1
    if len(sys.argv) > arg_id:
        actor_id = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1

    if actor_id >= len(actors):
        print 'Invalid actor_id: ', actor_id
        sys.exit()

    actor = actors[actor_id]
    sequences = sequences[actor]

    if seq_id >= len(sequences):
        print 'Invalid dataset_id: ', seq_id
        sys.exit()

    seq_name = sequences[seq_id]

    print 'actor: ', actor
    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name

    ground_truth_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)

    no_of_frames, corner_size = np.shape(ground_truth)
    print 'no_of_frames: ', no_of_frames
    print 'corner_size: ', corner_size

    end_id = no_of_frames

    init_corners = np.asarray([ground_truth[0, 0:2].tolist(),
                               ground_truth[0, 2:4].tolist(),
                               ground_truth[0, 4:6].tolist(),
                               ground_truth[0, 6:8].tolist()]).T

    std_pts, std_corners = getNormalizedUnitSquarePts(50, 50, 0.5)
    init_hom_mat = np.mat(util.compute_homography(std_corners, init_corners))
    init_hom_mat_inv = np.linalg.inv(init_hom_mat)

    proc_times = []

    curr_delta_state = np.zeros([end_id - start_id + 1, 8])

    prev_corners = init_corners
    prev_hom_mat = init_hom_mat

    line_id = 0

    for frame_id in xrange(start_id, end_id):
        curr_corners = np.asarray([ground_truth[frame_id, 0:2].tolist(),
                                   ground_truth[frame_id, 2:4].tolist(),
                                   ground_truth[frame_id, 4:6].tolist(),
                                   ground_truth[frame_id, 6:8].tolist()]).T
        try:
            curr_hom_mat = np.mat(util.compute_homography(prev_corners, curr_corners))
        except np.linalg.linalg.LinAlgError as error_msg:
            print'Error encountered while computing homography for frame {:d}: {:s}'.format(frame_id, error_msg)
            break

        curr_delta_hom_mat = np.linalg.inv(prev_hom_mat) * curr_hom_mat * prev_hom_mat
        curr_delta_hom_mat /= curr_delta_hom_mat[2, 2]

        curr_delta_state[line_id, 0] = curr_delta_hom_mat[0, 0] - 1
        curr_delta_state[line_id, 1] = curr_delta_hom_mat[0, 1]
        curr_delta_state[line_id, 2] = curr_delta_hom_mat[0, 2]
        curr_delta_state[line_id, 3] = curr_delta_hom_mat[1, 0]
        curr_delta_state[line_id, 4] = curr_delta_hom_mat[1, 1] - 1
        curr_delta_state[line_id, 5] = curr_delta_hom_mat[1, 2]
        curr_delta_state[line_id, 6] = curr_delta_hom_mat[2, 0]
        curr_delta_state[line_id, 7] = curr_delta_hom_mat[2, 1]

        prev_corners = np.copy(curr_corners)
        prev_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners))

        line_id += 1

    out_dir = out_root_dir + '/' + actor
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = out_dir + '/' + seq_name + '.txt'
    out_data = np.vstack((np.mean(curr_delta_state, axis=0), np.std(curr_delta_state, axis=0), curr_delta_state))
    np.savetxt(out_path, out_data, fmt='%f', delimiter=',')
    print 'Exiting....'




