from Misc import *
import time
import os

if __name__ == '__main__':

    db_root_dir = '../Datasets'
    img_root_dir = '../Image Data'

    params_dict = getParamDict()
    param_ids = readDistGridParams()

    actors = params_dict['actors']
    sequences = params_dict['sequences']
    challenges = params_dict['challenges']
    filter_types = params_dict['filter_types']

    actor_id = param_ids['actor_id']
    seq_id = param_ids['seq_id']
    challenge_id = param_ids['challenge_id']
    filter_id = param_ids['filter_id']
    kernel_size = param_ids['kernel_size']

    dof = 8

    n_bins = param_ids['n_bins']
    # dof = param_ids['dof']
    grid_res = param_ids['grid_res']
    show_img = param_ids['show_img']

    n_interp = 9

    arg_id = 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        appearance_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        opt_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        selective_opt = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        filter_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        inc_type = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        write_track_data = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        read_dist_data = int(sys.argv[arg_id])
        arg_id += 1

    if actor_id >= len(actors):
        print 'Invalid actor_id: ', actor_id
        sys.exit()

    actor = actors[actor_id]
    sequences = sequences[actor]

    if seq_id >= len(sequences):
        print 'Invalid seq_id: ', seq_id
        sys.exit()
    if challenge_id >= len(challenges):
        print 'Invalid challenge_id: ', challenge_id
        sys.exit()
    if filter_id >= len(filter_types):
        print 'Invalid filter_id: ', filter_id
        sys.exit()

    seq_name = sequences[seq_id]
    filter_type = filter_types[filter_id]
    challenge = challenges[challenge_id]

    if actor == 'METAIO':
        seq_name = seq_name + '_' + challenge

    print 'actor: ', actor
    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    print 'filter_type: ', filter_type
    print 'kernel_size: ', kernel_size

    src_folder = db_root_dir + '/' + actor + '/' + seq_name
    dst_folder = src_folder + '_interp' + str(n_interp + 1)

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    file_list = os.listdir(src_folder)
    no_of_frames = len(file_list)
    print 'no_of_frames: ', no_of_frames
    end_id = no_of_frames

    out_frame_id = 0
    start_id = 0

    init_img = final_img = None

    for frame_id in xrange(start_id, end_id - 1):

        print '\n\nframe_id: ', frame_id, ' of ', end_id

        init_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id + 1))
        final_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id + 2))
        interpolated_images = getLinearInterpolatedImages(init_img, final_img, n_interp)

        out_fname = dst_folder + '/frame{:05d}.jpg'.format(out_frame_id + 1)
        out_frame_id += 1
        cv2.imwrite(out_fname, init_img)
        for i in xrange(n_interp):
            out_fname = dst_folder + '/frame{:05d}.jpg'.format(out_frame_id + 1)
            out_frame_id += 1
            cv2.imwrite(out_fname, interpolated_images[i])

    out_fname = dst_folder + '/frame{:05d}.jpg'.format(out_frame_id + 1)
    out_frame_id += 1
    cv2.imwrite(out_fname, final_img)






