import distanceUtils
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
    tracker_types = params_dict['params_dict']
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
    mi_bins = dist_prarams['mi_bins']
    scv_bins = dist_prarams['scv_bins']
    ccre_bins = dist_prarams['ccre_bins']

    write_img_data = 0

    std_resx = 50
    std_resy = 50
    n_pts = std_resx * std_resy
    grid_res = 80

    using_mi = 0
    using_scv = 0
    using_ncc = 0
    using_ccre = 0

    tx_res, ty_res = [grid_res, grid_res]
    theta_res, scale_res = [grid_res, grid_res]
    a_res, b_res = [grid_res, grid_res]
    v1_res, v2_res = [grid_res, grid_res]

    trans_thr = 1
    theta_thresh, scale_thresh = [np.pi / 32, 0.1]
    a_thresh, b_thresh = [1, 1]
    v1_thresh, v2_thresh = [1, 1]

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
        filter_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        tracker_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        inc_type = sys.argv[arg_id]
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

    if appearance_model == 'ssd':
        dist_func = getSSDPoints
    elif appearance_model == 'mi':
        distanceUtils.initStateVars(n_pts, mi_bins)
        dist_func = distanceUtils.getMIPoints
        using_mi = 1
    elif appearance_model == 'mi2':
        distanceUtils.initStateVars(n_pts, mi_bins)
        dist_func = distanceUtils.getMIPoints2
        using_mi = 1
    elif appearance_model == 'mi_new':
        distanceUtils.initStateVars(n_pts, mi_bins)
        dist_func = distanceUtils.getMIPointsNew
        using_mi = 1
    elif appearance_model == 'ncc':
        dist_func = getNCCPoints
        using_ncc = 1
    elif appearance_model == 'ncc2':
        dist_func = getNCCPoints2
    elif appearance_model == 'scv':
        distanceUtils.initStateVars(n_pts, scv_bins)
        dist_func = distanceUtils.getSCVPoints
        using_scv = 1
    elif appearance_model == 'scv2':
        distanceUtils.initStateVars(n_pts, scv_bins)
        dist_func = distanceUtils.getSCVPoints2
        using_scv = 1
    elif appearance_model == 'ccre':
        distanceUtils.initStateVars(n_pts, ccre_bins)
        dist_func = distanceUtils.getCCREPoints
        using_ccre = 1
    else:
        raise StandardError('Invalid appearance model: {:s}'.format(appearance_model))

    tx_min, tx_max = [-trans_thr, trans_thr]
    ty_min, ty_max = [-trans_thr, trans_thr]
    theta_min, theta_max = [-theta_thresh, theta_thresh]
    scale_min, scale_max = [-scale_thresh, scale_thresh]
    a_min, a_max = [-a_thresh, a_thresh]
    b_min, b_max = [-b_thresh, b_thresh]
    v1_min, v1_max = [-v1_thresh, v1_thresh]
    v2_min, v2_max = [-v2_thresh, v2_thresh]

    tx_vec = np.linspace(tx_min, tx_max, tx_res)
    ty_vec = np.linspace(ty_min, ty_max, ty_res)
    theta_vec = np.linspace(theta_min, theta_max, theta_res)
    scale_vec = np.linspace(scale_min, scale_max, scale_res)
    a_vec = np.linspace(a_min, a_max, a_res)
    b_vec = np.linspace(b_min, b_max, b_res)
    v1_vec = np.linspace(v1_min, v1_max, v1_res)
    v2_vec = np.linspace(v2_min, v2_max, v2_res)

    tx_vec = np.insert(tx_vec, np.argwhere(tx_vec >= 0)[0, 0], 0).astype(np.float64)
    ty_vec = np.insert(ty_vec, np.argwhere(ty_vec >= 0)[0, 0], 0).astype(np.float64)
    theta_vec = np.insert(theta_vec, np.argwhere(theta_vec >= 0)[0, 0], 0).astype(np.float64)
    scale_vec = np.insert(scale_vec, np.argwhere(scale_vec >= 0)[0, 0], 0).astype(np.float64)
    a_vec = np.insert(a_vec, np.argwhere(a_vec >= 0)[0, 0], 0).astype(np.float64)
    b_vec = np.insert(b_vec, np.argwhere(b_vec >= 0)[0, 0], 0).astype(np.float64)
    v1_vec = np.insert(v1_vec, np.argwhere(v1_vec >= 0)[0, 0], 0).astype(np.float64)
    v2_vec = np.insert(v2_vec, np.argwhere(v2_vec >= 0)[0, 0], 0).astype(np.float64)

    if tracker_type != 'gt':
        in_fname = 'Tracking Result/{:s}_{:s}_params.txt'.format(seq_name, tracker_type)
        out_fname = 'Tracking Result/{:s}_{:s}_params_inv.txt'.format(seq_name, tracker_type)
        dec_params_mat = reformatHomFile(in_fname, out_fname)

    src_folder = db_root_path + '/' + actor + '/' + seq_name

    img_folder = 'Image Data'
    if filter_type != 'none':
        img_fname = img_folder + '/' + seq_name + '_' + filter_type + str(kernel_size) + '.bin'
        dist_folder = 'Distance Data/' + seq_name + '_' + appearance_model + '_' + tracker_type + '_' + filter_type + str(
            kernel_size)
    else:
        img_fname = img_folder + '/' + seq_name + '.bin'
        dist_folder = 'Distance Data/' + seq_name + '_' + appearance_model + '_' + tracker_type

    if not os.path.exists(dist_folder):
        os.makedirs(dist_folder)

    dist_fname = dist_folder + '/' + inc_type + '_' + grid_type + '.bin'

    if os.path.isfile(dist_fname):
        s = raw_input('\nWarning: The distance file already exists. Proceed with overwrite ?\n')
        if s == 'n' or s == 'N':
            sys.exit()
    dist_fid = open(dist_fname, 'wb')

    if write_img_data:
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        img_fid = open(img_fname, 'wb')

    if tracker_type != 'gt':
        ground_truth_fname = db_root_path + '/' + actor + '/' + seq_name + '_' + tracker_type + '.txt'
    else:
        ground_truth_fname = db_root_path + '/' + actor + '/' + seq_name + '.txt'

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
    if using_mi:
        init_img_gs = init_img_gs * (mi_bins - 1) / 255.0
    elif using_scv:
        init_img_gs = init_img_gs * (scv_bins - 1) / 255.0
    elif using_ccre:
        init_img_gs = init_img_gs * (ccre_bins - 1) / 255.0
    elif using_ncc:
        init_img_gs = init_img_gs - np.mean(init_img_gs)

    if write_img_data:
        init_img_gs.astype(np.uint8).tofile(img_folder + '/' + 'frame_0_gs.bin')
    # k=util.bilin_interp(init_img_gs, init_pts[0, 0], init_pts[1, 0])
    # print 'k: ', k
    init_pixel_vals = np.array([util.bilin_interp(init_img_gs, init_pts[0, pt_id], init_pts[1, pt_id]) for pt_id in
                                xrange(n_pts)])
    if distanceUtils.isInited():
        print 'using distanceUtils module'
        init_pixel_vals = np.rint(init_pixel_vals)
    # cv2.imshow('Init Image', init_img)

    if grid_type == 'trans':
        y_vec = tx_vec
        x_vec = ty_vec
        print 'tx_vec: ', tx_vec
        print 'ty_vec: ', ty_vec
    elif grid_type == 'rtx':
        y_vec = tx_vec
        x_vec = theta_vec
        print 'tx_vec: ', tx_vec
        print 'theta_vec: ', theta_vec
    elif grid_type == 'rty':
        y_vec = ty_vec
        x_vec = theta_vec
        print 'ty_vec: ', ty_vec
        print 'theta_vec: ', theta_vec
    elif grid_type == 'rs':
        y_vec = scale_vec
        x_vec = theta_vec
        print 'scale_vec: ', scale_vec
        print 'theta_vec: ', theta_vec
    elif grid_type == 'shear':
        y_vec = a_vec
        x_vec = b_vec
        print 'a_vec: ', a_vec
        print 'b_vec: ', b_vec
    elif grid_type == 'proj':
        y_vec = v1_vec
        x_vec = v2_vec
        print 'v1_vec: ', v1_vec
        print 'v2_vec: ', v2_vec
    else:
        raise StandardError('Invalid grid_type: ' + grid_type)

    np.array([start_id - 1, start_id, x_vec.size, y_vec.size], dtype=np.uint32).tofile(dist_fid)
    x_vec.tofile(dist_fid)
    y_vec.tofile(dist_fid)
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
        if filter_type != 'none':
            curr_img_gs = applyFilter(curr_img_gs, filter_type, kernel_size)
        if using_mi:
            curr_img_gs = curr_img_gs * (mi_bins - 1) / 255.0
        elif using_scv:
            curr_img_gs = curr_img_gs * (scv_bins - 1) / 255.0
        elif using_ccre:
            curr_img_gs = curr_img_gs * (ccre_bins - 1) / 255.0
        elif using_ncc:
            curr_img_gs = curr_img_gs - np.mean(curr_img_gs)

        # np.savetxt('curr_img_gs_' + str(frame_id) + '.txt', curr_img_gs, fmt='%12.9f')

        curr_pixel_vals = np.mat([util.bilin_interp(curr_img_gs, curr_pts[0, pt_id], curr_pts[1, pt_id]) for pt_id in
                                  xrange(n_pts)])
        if distanceUtils.isInited():
            curr_pixel_vals = np.rint(curr_pixel_vals)
        # np.savetxt('curr_pixel_vals_' + str(frame_id) + '.txt', curr_pixel_vals, fmt='%3d')


        # drawRegion(curr_img, curr_corners, (0, 255, 0))
        # cv2.imshow('Current Image', curr_img)
        # cv2.waitKey(100)

        start_time = time.clock()

        if grid_type == 'trans':
            if inc_type == 'fa' or inc_type == 'fc':
                dist_grid = getTransDistanceGridPre(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs, init_pixel_vals,
                                                    tx_vec, ty_vec, dist_func)
            elif inc_type == 'ia' or inc_type == 'ic':
                dist_grid = getTransDistanceGridPre(std_pts, init_hom_mat, init_norm_mat, init_img_gs, curr_pixel_vals,
                                                    tx_vec, ty_vec, dist_func)
            else:
                raise SyntaxError('Invalid increment type: ' + inc_type)

        elif grid_type == 'rtx':
            if inc_type == 'fa':
                dist_grid = getRTxDistanceGridAdd(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs, init_pixel_vals,
                                                  theta_vec, tx_vec, dist_func)
            elif inc_type == 'ia':
                dist_grid = getRTxDistanceGridAdd(std_pts, init_hom_mat, init_norm_mat, init_img_gs, curr_pixel_vals,
                                                  theta_vec, tx_vec, dist_func)
            elif inc_type == 'ic':
                dist_grid = getRTxDistanceGridComp(std_pts, init_hom_mat, init_norm_mat, init_img_gs, curr_pixel_vals,
                                                   theta_vec, tx_vec, dist_func)
            elif inc_type == 'fc':
                dist_grid = getRTxDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs, init_pixel_vals,
                                                   theta_vec, tx_vec, dist_func)
            else:
                raise SyntaxError('Invalid increment type: ' + inc_type)

        elif grid_type == 'rty':
            if inc_type == 'fa':
                dist_grid = getRTyDistanceGridAdd(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs, init_pixel_vals,
                                                  theta_vec, ty_vec, dist_func)
            elif inc_type == 'ia':
                dist_grid = getRTyDistanceGridAdd(std_pts, init_hom_mat, init_norm_mat, init_img_gs, curr_pixel_vals,
                                                  theta_vec, ty_vec, dist_func)
            elif inc_type == 'ic':
                dist_grid = getRTyDistanceGridComp(std_pts, init_hom_mat, init_norm_mat, init_img_gs, curr_pixel_vals,
                                                   theta_vec, ty_vec, dist_func)
            elif inc_type == 'fc':
                dist_grid = getRTyDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs, init_pixel_vals,
                                                   theta_vec, ty_vec, dist_func)
            else:
                raise SyntaxError('Invalid increment type: ' + inc_type)

        elif grid_type == 'rs':
            if inc_type == 'fa':
                dist_grid = getRSDistanceGridAdd(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs, init_pixel_vals,
                                                 scale_vec, theta_vec, dist_func)
            elif inc_type == 'ia':
                dist_grid = getRSDistanceGridAdd(std_pts, init_hom_mat, init_norm_mat, init_img_gs, curr_pixel_vals,
                                                 scale_vec, theta_vec, dist_func)
            elif inc_type == 'ic':
                dist_grid = getRSDistanceGridComp(std_pts, init_hom_mat, init_norm_mat, init_img_gs, curr_pixel_vals,
                                                  scale_vec, theta_vec, dist_func)
            elif inc_type == 'fc':
                dist_grid = getRSDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs, init_pixel_vals,
                                                  scale_vec, theta_vec, dist_func)
            else:
                raise SyntaxError('Invalid increment type: ' + inc_type)

        elif grid_type == 'shear':
            if inc_type == 'fa':
                dist_grid = getShearDistanceGridAdd(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs, init_pixel_vals,
                                                    a_vec, b_vec, dist_func)
            elif inc_type == 'ia':
                dist_grid = getShearDistanceGridAdd(std_pts, init_hom_mat, init_norm_mat, init_img_gs, curr_pixel_vals,
                                                    a_vec, b_vec, dist_func)
            elif inc_type == 'ic':
                dist_grid = getShearDistanceGridComp(std_pts, init_hom_mat, init_norm_mat, init_img_gs, curr_pixel_vals,
                                                     a_vec, b_vec, dist_func)
            elif inc_type == 'fc':
                dist_grid = getShearDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs, init_pixel_vals,
                                                     a_vec, b_vec, dist_func)
            else:
                raise SyntaxError('Invalid increment type: ' + inc_type)

        elif grid_type == 'proj':
            if inc_type == 'fa':
                dist_grid = getProjDistanceGridAdd(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs, init_pixel_vals,
                                                   v1_vec, v2_vec, dist_func)
            elif inc_type == 'ia':
                dist_grid = getProjDistanceGridAdd(std_pts, init_hom_mat, init_norm_mat, init_img_gs, curr_pixel_vals,
                                                   v1_vec, v2_vec, dist_func)
            elif inc_type == 'ic':
                dist_grid = getProjDistanceGridComp(std_pts, init_hom_mat, init_norm_mat, init_img_gs, curr_pixel_vals,
                                                    v1_vec, v2_vec, dist_func)
            elif inc_type == 'fc':
                dist_grid = getProjDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs, init_pixel_vals,
                                                    v1_vec, v2_vec, dist_func)
            else:
                raise SyntaxError('Invalid increment type: ' + inc_type)
        else:
            raise SyntaxError('Invalid grid type: ' + grid_type)
        end_time = time.clock()
        curr_time = end_time - start_time
        curr_fps = 1.0 / curr_time
        proc_times.append(curr_time)

        print 'frame_id:\t{:-5d}\tTime:\t{:-14.10f}'.format(frame_id, curr_time)

        # dist_file_bin = open(dist_folder + '/' + grid_type + '_dist_grid_' + str(frame_id) + '.bin', 'wb')
        dist_grid.tofile(dist_fid)
        current_offset = dist_fid.tell()
        dist_fid.seek(0)
        np.array([frame_id], dtype=np.uint32).tofile(dist_fid)
        dist_fid.seek(current_offset)
        # dist_grid.tofile(dist_folder + '/' + 'dist_grid_' + str(frame_id) + '.bin')

        # img_file_bin = open(dist_folder + '/' + 'frame_' + str(frame_id) + '_gs.bin', 'wb')
        if write_img_data:
            curr_img_gs.astype(np.uint8).tofile(img_fid)

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
    if write_img_data:
        img_fid.close()
    dist_fid.close()
