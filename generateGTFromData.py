from distanceGrid import *
import time
import os
from Misc import getParamDict
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
    inc_types = params_dict['inc_types']
    appearance_models = params_dict['appearance_models']
    challenges = params_dict['challenges']

    actor_id = param_ids['actor_id']
    seq_id = param_ids['seq_id']
    inc_id = param_ids['inc_id']
    appearance_id = param_ids['appearance_id']
    n_bins = param_ids['n_bins']
    challenge_id = param_ids['challenge_id']

    write_img_data = 0

    std_resx = 50
    std_resy = 50
    n_pts = std_resx * std_resy
    grid_res = 9
    grid_thresh = 0.05
    start_id = 1

    tx_res, ty_res = [grid_res, grid_res]
    theta_res, scale_res = [grid_res, grid_res]
    a_res, b_res = [grid_res, grid_res]
    v1_res, v2_res = [grid_res, grid_res]

    trans_thr = grid_thresh
    theta_thresh, scale_thresh = [grid_thresh, grid_thresh]
    a_thresh, b_thresh = [grid_thresh, grid_thresh]
    v1_thresh, v2_thresh = [grid_thresh, grid_thresh]

    arg_id = 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        inc_type = sys.argv[arg_id]
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
    inc_type = inc_types[inc_id]
    appearance_model = appearance_models[appearance_id]
    challenge = challenges[challenge_id]

    if actor == 'METAIO':
        seq_name = seq_name + '_' + challenge

    print 'actor: ', actor
    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    print 'inc_type: ', inc_type
    print 'appearance_model: ', appearance_model
    print 'n_bins: ', n_bins

    dist_func, pre_proc_func, post_proc_func, opt_func = getDistanceFunction(appearance_model, n_pts, n_bins)

    tx_min, tx_max = [-trans_thr, trans_thr]
    ty_min, ty_max = [-trans_thr, trans_thr]
    theta_min, theta_max = [-theta_thresh, theta_thresh]
    scale_min, scale_max = [-scale_thresh, scale_thresh]
    a_min, a_max = [-a_thresh, a_thresh]
    b_min, b_max = [-b_thresh, b_thresh]
    v1_min, v1_max = [-v1_thresh, v1_thresh]
    v2_min, v2_max = [-v2_thresh, v2_thresh]

    vec_min = [tx_min, ty_min, theta_min, scale_min, a_min, b_min, v1_min, v2_min]
    vec_max = [tx_max, ty_max, theta_max, scale_max, a_max, b_max, v1_max, v2_max]
    vec_res = [tx_res, ty_res, theta_res, scale_res, a_res, b_res, v1_res, v2_res]
    tx_vec, ty_vec, theta_vec, scale_vec, a_vec, b_vec, v1_vec, v2_vec, tx2_vec, ty2_vec = getGridVectors(vec_min,
                                                                                                          vec_max,
                                                                                                          vec_res)

    if not os.path.exists(track_root_dir):
        os.mkdir(track_root_dir)

    track_data_dir = '{:s}/{:s}_{:s}'.format(track_root_dir, appearance_model, seq_name)
    if not os.path.exists(track_data_dir):
        os.mkdir(track_data_dir)

    dist_dir = dist_root_dir + '/' + appearance_model + '_' + seq_name

    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)

    dist_fname = dist_dir + '/' + 'brute_hom_' + inc_type + '.bin'
    if os.path.isfile(dist_fname):
        s = raw_input('\nWarning: The distance file already exists. Proceed with overwrite ?\n')
        if s == 'n' or s == 'N':
            sys.exit()
    dist_fid = open(dist_fname, 'wb')

    src_folder = db_root_dir + '/' + actor + '/' + seq_name
    ground_truth_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)
    file_list = os.listdir(src_folder)
    print 'file_list: ', file_list

    no_of_frames = len(file_list)
    print 'no_of_frames: ', no_of_frames

    end_id = no_of_frames

    init_corners = np.asarray([ground_truth[0, 0:2].tolist(),
                               ground_truth[0, 2:4].tolist(),
                               ground_truth[0, 4:6].tolist(),
                               ground_truth[0, 6:8].tolist()]).T

    std_pts, std_corners = getNormalizedUnitSquarePts(std_resx, std_resy, 0.5)
    std_pts_hm = util.homogenize(std_pts)
    (init_corners_norm, init_norm_mat) = getNormalizedPoints(init_corners)
    init_hom_mat = np.mat(util.compute_homography(std_corners, init_corners_norm))
    init_pts_norm = util.dehomogenize(init_hom_mat * std_pts_hm)
    init_pts = util.dehomogenize(init_norm_mat * util.homogenize(init_pts_norm))

    # ret, init_img = cap.read()
    init_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(1))
    if init_img is None:
        raise StandardError(
            'The initial image could not be read from the file:\n' + src_folder + '/frame{:05d}.jpg'.format(1))
    init_img_gs = cv2.cvtColor(init_img, cv2.cv.CV_BGR2GRAY).astype(np.float64)
    init_img_gs = pre_proc_func(init_img_gs)
    if write_img_data:
        init_img_gs.astype(np.uint8).tofile(img_root_dir + '/' + 'frame_0_gs.bin')

    init_pixel_vals = np.array([util.bilin_interp(init_img_gs, init_pts[0, pt_id], init_pts[1, pt_id]) for pt_id in
                                xrange(n_pts)])
    init_pixel_vals = post_proc_func(init_pixel_vals)
    # cv2.imshow('Init Image', init_img)

    if start_id > 1:
        curr_corners = np.asarray([ground_truth[start_id - 1, 0:2].tolist(),
                                   ground_truth[start_id - 1, 2:4].tolist(),
                                   ground_truth[start_id - 1, 4:6].tolist(),
                                   ground_truth[start_id - 1, 6:8].tolist()]).T

        (curr_corners_norm, curr_norm_mat) = getNormalizedPoints(curr_corners)
        curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners_norm))
        curr_pts_norm = util.dehomogenize(curr_hom_mat * std_pts_hm)
        curr_pts = util.dehomogenize(curr_norm_mat * util.homogenize(curr_pts_norm))
    else:
        curr_corners = init_corners
        curr_norm_mat = init_norm_mat
        curr_hom_mat = init_hom_mat
        curr_pts = init_pts

    proc_times = []

    for frame_id in xrange(start_id, end_id):
        # ret, curr_img = cap.read()
        curr_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id + 1))
        # print 'curr_img: ', curr_img
        if curr_img is None:
            break
        curr_img_gs = cv2.cvtColor(curr_img, cv2.cv.CV_BGR2GRAY).astype(np.float64)
        curr_img_gs = pre_proc_func(curr_img_gs)

        # np.savetxt('curr_img_gs_' + str(frame_id) + '.txt', curr_img_gs, fmt='%12.9f')

        curr_pixel_vals = np.mat([util.bilin_interp(curr_img_gs, curr_pts[0, pt_id], curr_pts[1, pt_id]) for pt_id in
                                  xrange(n_pts)])
        curr_pixel_vals = post_proc_func(curr_pixel_vals)
        start_time = time.clock()

        opt_params, dist_grid = getHomDistanceGridPre(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs,
                                                      init_pixel_vals,
                                                      tx_vec, ty_vec, theta_vec, scale_vec, a_vec, b_vec, v1_vec,
                                                      v2_vec,
                                                      dist_func)
        end_time = time.clock()
        curr_time = end_time - start_time
        dist_grid.tofile(dist_fid)
        current_offset = dist_fid.tell()
        dist_fid.seek(0)
        np.array([frame_id], dtype=np.uint32).tofile(dist_fid)
        dist_fid.seek(current_offset)

        curr_corners = np.asarray([ground_truth[frame_id, 0:2].tolist(),
                                   ground_truth[frame_id, 2:4].tolist(),
                                   ground_truth[frame_id, 4:6].tolist(),
                                   ground_truth[frame_id, 6:8].tolist()]).T

        curr_corners_norm, curr_norm_mat = getNormalizedPoints(curr_corners)
        try:
            curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners_norm))
        except np.linalg.linalg.LinAlgError as error_msg:
            print'Error encountered while computing homography for frame {:d}: {:s}'.format(frame_id, error_msg)
            break
        curr_pts_norm = util.dehomogenize(curr_hom_mat * std_pts_hm)
        curr_pts = util.dehomogenize(curr_norm_mat * util.homogenize(curr_pts_norm))

        if distanceUtils.isInited():
            distanceUtils.freeStateVars()

        np.array(proc_times, dtype=np.float64).tofile(dist_fid)
        dist_fid.close()
        print 'Exiting....'




