# from DecompUtils import *
# from distanceGrid import applyFilter
# import time
import os
import cv2
import numpy as np
# from Misc import getParamDict

if __name__ == '__main__':
    db_root_dir = 'C:/Datasets'
    track_root_dir = '../Tracking Data'
    img_root_dir = '../Image Data'
    dist_root_dir = '../Distance Data'
    track_img_root_dir = '../Tracked Images'

    # params_dict = getParamDict()
    # param_ids = readDistGridParams()
    #
    # actors = params_dict['actors']
    # sequences = params_dict['sequences']
    # challenges = params_dict['challenges']
    # filter_types = params_dict['filter_types']
    #
    # actor_id = param_ids['actor_id']
    # seq_id = param_ids['seq_id']
    # challenge_id = param_ids['challenge_id']
    # inc_id = param_ids['inc_id']
    # start_id = param_ids['start_id']
    # filter_id = param_ids['filter_id']
    # kernel_size = param_ids['kernel_size']
    # show_img = param_ids['show_img']
    #
    # arg_id = 1
    # if len(sys.argv) > arg_id:
    #     actor_id = int(sys.argv[arg_id])
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     seq_id = int(sys.argv[arg_id])
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     challenge_id = int(sys.argv[arg_id])
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     filter_id = int(sys.argv[arg_id])
    #     arg_id += 1
    #
    # if actor_id >= len(actors):
    #     print 'Invalid actor_id: ', actor_id
    #     sys.exit()
    #
    # actor = actors[actor_id]
    # sequences = sequences[actor]
    #
    # if seq_id >= len(sequences):
    #     print 'Invalid dataset_id: ', seq_id
    #     sys.exit()
    # if challenge_id >= len(challenges):
    #     print 'Invalid challenge_id: ', challenge_id
    #     sys.exit()
    # if filter_id >= len(filter_types):
    #     print 'Invalid filter_id: ', filter_id
    #     sys.exit()
    #
    # seq_name = sequences[seq_id]
    # filter_type = filter_types[filter_id]
    # challenge = challenges[challenge_id]
    #
    # if actor == 'METAIO':
    #     seq_name = seq_name + '_' + challenge

    actor = 'GRAM'
    seq_name = 'idot_1_intersection_city_day_short'
    start_id = 0
    filter_type = 'none'
    kernel_size = 3
    show_img = 1

    print 'actor: ', actor
    # print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    # print 'filter_type: ', filter_type
    # print 'kernel_size: ', kernel_size

    src_dir = db_root_dir + '/' + actor + '/Images/' + seq_name

    if not os.path.exists(img_root_dir):
        os.makedirs(img_root_dir)
    if filter_type != 'none':
        img_fname = img_root_dir + '/' + seq_name + '_' + filter_type + str(kernel_size) + '.bin'
    else:
        img_fname = img_root_dir + '/' + seq_name + '.bin'

    print 'Reading images from: {:s}'.format(src_dir)
    print 'Writing image binary data to: {:s}'.format(img_fname)

    img_fid = open(img_fname, 'wb')
    file_list = os.listdir(src_dir)
    # print 'file_list: ', file_list
    no_of_frames = len(file_list)
    print 'no_of_frames: ', no_of_frames
    end_id = no_of_frames

    init_img = cv2.imread(src_dir + '/image{:06d}.jpg'.format(1))
    img_height = init_img.shape[0]
    img_width = init_img.shape[1]

    # np.array([no_of_frames, ], dtype=np.uint32).tofile(img_fid)
    np.array([img_width, img_height], dtype=np.uint32).tofile(img_fid)

    win_name = 'Filtered Image'
    if show_img:
        cv2.namedWindow(win_name)

    for frame_id in xrange(start_id, end_id):
        # print 'frame_id: ', frame_id
        curr_img = cv2.imread(src_dir + '/image{:06d}.jpg'.format(frame_id + 1))
        if len(curr_img.shape) == 3:
            curr_img_gs = cv2.cvtColor(curr_img, cv2.cv.CV_BGR2GRAY)
        else:
            curr_img_gs = curr_img
        # if filter_type != 'none':
        #     curr_img_gs = applyFilter(curr_img_gs, filter_type, kernel_size)
        curr_img_gs.astype(np.uint8).tofile(img_fid)
        if show_img:
            cv2.imshow(win_name, curr_img_gs)
            if cv2.waitKey(1) == 27:
                break

    img_fid.close()
