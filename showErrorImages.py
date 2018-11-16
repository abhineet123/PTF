from DecompUtils import *
import utility as util
import time
import os
from Misc import *
import shutil
from ImageUtils import draw_region


def updateParams(x):
    global err_thresh, err_pix_vals, thresh_err_img, std_resx, std_resy
    err_thresh = x
    thresh_pix_vals = np.copy(err_pix_vals)
    thresh_pix_vals[thresh_pix_vals > err_thresh] = 255
    thresh_pix_vals[thresh_pix_vals < err_thresh] = 0
    thresh_err_img = np.reshape(thresh_pix_vals, (std_resx, std_resy)).astype(np.uint8)
    cv2.imshow('Thresholded Error Patch', thresh_err_img)


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

    write_img_data = 0

    std_resx = 200
    std_resy = 200
    n_pts = std_resx * std_resy
    grid_res = 100
    grid_thresh = 0.25

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
    img_fname = img_root_dir + '/' + seq_name

    if filter_type != 'none':
        img_fname = img_fname + '_' + filter_type + str(kernel_size) + '.bin'
    img_fname += '.bin'

    if write_img_data:
        if not os.path.exists(img_root_dir):
            os.makedirs(img_root_dir)
        img_fid = open(img_fname, 'wb')

    ground_truth_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'

    ground_truth = readTrackingData(ground_truth_fname)
    no_of_frames = ground_truth.shape[0]
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
    if filter_type != 'none':
        init_img_gs = applyFilter(init_img_gs, filter_type, kernel_size)
    if write_img_data:
        init_img_gs.astype(np.uint8).tofile(img_root_dir + '/' + 'frame_0_gs.bin')

    init_pixel_vals = np.array([util.bilin_interp(init_img_gs, init_pts[0, pt_id], init_pts[1, pt_id]) for pt_id in
                                xrange(n_pts)])
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

    cv2.namedWindow('Initial Patch')
    cv2.namedWindow('Current Patch')
    cv2.namedWindow('Error Patch')

    init_patch = np.reshape(init_pixel_vals, (std_resx, std_resy)).astype(np.uint8)
    cv2.imshow('Initial Patch', init_patch)

    proc_times = []
    pause_exec = 1

    curr_img_win = 'Current Image'
    cv2.namedWindow(curr_img_win)

    err_thresh = 35
    err_max = 255
    cv2.createTrackbar('err_thresh', curr_img_win, err_thresh, err_max, updateParams)

    frame_id = 1
    while frame_id < end_id:

        curr_corners = np.asarray([ground_truth[frame_id, 0:2].tolist(),
                                   ground_truth[frame_id, 2:4].tolist(),
                                   ground_truth[frame_id, 4:6].tolist(),
                                   ground_truth[frame_id, 6:8].tolist()]).T

        # curr_corners_norm, curr_norm_mat = getNormalizedPoints(curr_corners)
        try:
            curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners))
        except np.linalg.linalg.LinAlgError as error_msg:
            print'Error encountered while computing homography for frame {:d}: {:s}'.format(frame_id, error_msg)
            break
        # curr_pts_norm = util.dehomogenize(curr_hom_mat * std_pts_hm)
        # curr_pts = util.dehomogenize(curr_norm_mat * util.homogenize(curr_pts_norm))

        curr_pts = util.dehomogenize(curr_hom_mat * std_pts_hm)

        # ret, curr_img = cap.read()
        curr_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id + 1))
        # print 'curr_img: ', curr_img
        if curr_img is None:
            break
        curr_img_gs = cv2.cvtColor(curr_img, cv2.cv.CV_BGR2GRAY).astype(np.float64)
        if filter_type != 'none':
            curr_img_gs = applyFilter(curr_img_gs, filter_type, kernel_size)


        # np.savetxt('curr_img_gs_' + str(frame_id) + '.txt', curr_img_gs, fmt='%12.9f')

        curr_pixel_vals = np.mat([util.bilin_interp(curr_img_gs, curr_pts[0, pt_id], curr_pts[1, pt_id]) for pt_id in
                                  xrange(n_pts)])
        curr_patch = np.reshape(curr_pixel_vals, (std_resx, std_resy)).astype(np.uint8)

        err_pix_vals = np.abs(curr_pixel_vals - init_pixel_vals)
        err_img = np.reshape(err_pix_vals, (std_resx, std_resy)).astype(np.uint8)

        thresh_pix_vals = np.copy(err_pix_vals)
        thresh_pix_vals[thresh_pix_vals > err_thresh] = 255
        thresh_pix_vals[thresh_pix_vals < err_thresh] = 0

        thresh_err_img = np.reshape(thresh_pix_vals, (std_resx, std_resy)).astype(np.uint8)

        cv2.imshow('Current Patch', curr_patch)
        cv2.imshow('Error Patch', err_img)
        cv2.imshow('Thresholded Error Patch', thresh_err_img)

        draw_region(curr_img, curr_corners, (0, 255, 255), 2)

        cv2.imshow(curr_img_win, curr_img)

        key = cv2.waitKey(1 - pause_exec)
        if key == 27:
            break
        elif key == 32:
            pause_exec = 1 - pause_exec
        # np.savetxt('curr_pixel_vals_' + str(frame_id) + '.txt', curr_pixel_vals, fmt='%3d')


        # drawRegion(curr_img, curr_corners, (0, 255, 0))
        # cv2.imshow('Current Image', curr_img)
        # cv2.waitKey(100)

        # start_time = time.clock()
        #
        # end_time = time.clock()
        # curr_time = end_time - start_time
        # curr_fps = 1.0 / curr_time
        # proc_times.append(curr_time)

        # print 'frame_id:\t{:-5d}\tTime:\t{:-14.10f}'.format(frame_id, curr_time)

        frame_id += 1
        if write_img_data:
            curr_img_gs.astype(np.uint8).tofile(img_fid)

    if write_img_data:
        img_fid.close()
    print 'Exiting....'
