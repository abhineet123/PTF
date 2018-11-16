from distanceGrid import *
import time
import os
from Misc import *
import itertools

if __name__ == '__main__':

    db_root_dir = '../Datasets'
    track_root_dir = '../Tracking Data'
    img_root_dir = '../Image Data'
    dist_root_dir = '../Distance Data'
    track_img_root_dir = '../Tracked Images'

    params_dict = getParamDict()
    param_ids = readDistGridParams()

    actors = params_dict['actors']
    sequences = params_dict['sequences']
    challenges = params_dict['challenges']
    grid_types = params_dict['grid_types']
    inc_types = params_dict['inc_types']
    appearance_models = params_dict['appearance_models']
    tracker_types = params_dict['tracker_types']
    filter_types = params_dict['filter_types']
    opt_types = params_dict['opt_types']

    actor_id = param_ids['actor_id']
    seq_id = param_ids['seq_id']
    challenge_id = param_ids['challenge_id']
    grid_id = param_ids['grid_id']
    appearance_id = param_ids['appearance_id']
    inc_id = param_ids['inc_id']
    start_id = param_ids['start_id']
    filter_id = param_ids['filter_id']
    kernel_size = param_ids['kernel_size']
    opt_id = param_ids['opt_id']
    selective_opt = param_ids['selective_opt']
    selective_thresh = param_ids['selective_thresh']
    dof = 8

    n_interp = 1

    n_bins = param_ids['n_bins']
    # dof = param_ids['dof']
    grid_res = param_ids['grid_res']
    show_img = param_ids['show_img']

    write_img_data = 0
    write_track_data = 1
    write_dist_data = 1
    read_dist_data = 0
    use_jaccard_error = 0

    write_img = 0
    show_gt = 1
    show_trans = 1
    show_rs = 1
    show_shear = 1
    show_proj = 1
    use_gt = 0

    show_gt = show_gt and show_img
    show_trans = show_trans and show_img
    show_rs = show_rs and show_img
    show_shear = show_shear and show_img
    show_proj = show_proj and show_img

    std_resx = 50
    std_resy = 50

    n_pts = std_resx * std_resy

    tx_res = grid_res
    ty_res = grid_res
    theta_res = grid_res
    scale_res = grid_res
    a_res = grid_res
    b_res = grid_res
    v1_res = grid_res
    v2_res = grid_res

    common_thresh = 0.1

    trans_thr = common_thresh
    theta_thresh, scale_thresh = [np.pi / 32, common_thresh]
    a_thresh, b_thresh = [common_thresh, common_thresh]
    v1_thresh, v2_thresh = [common_thresh, common_thresh]

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
    if opt_id >= len(opt_types):
        print 'Invalid opt_id: ', opt_id
        sys.exit()

    seq_name = sequences[seq_id]
    filter_type = filter_types[filter_id]
    opt_type = opt_types[opt_id]
    inc_type = inc_types[inc_id]
    appearance_model = appearance_models[appearance_id]
    challenge = challenges[challenge_id]

    if opt_type == 'ind':
        read_dist_data = 1

    if read_dist_data:
        write_dist_data = 0

    if actor == 'METAIO':
        seq_name = seq_name + '_' + challenge

    print 'actor: ', actor
    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    print 'inc_type: ', inc_type
    print 'appearance_model: ', appearance_model
    print 'filter_type: ', filter_type
    print 'kernel_size: ', kernel_size
    print 'opt_type: ', opt_type
    print 'selective_opt: ', selective_opt
    print 'read_dist_data: ', read_dist_data
    print 'write_dist_data: ', write_dist_data

    dist_func, pre_proc_func, post_proc_func, opt_func, is_better = getDistanceFunction(appearance_model, n_pts, n_bins)

    src_folder = db_root_dir + '/' + actor + '/' + seq_name
    ground_truth_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)

    file_list = os.listdir(src_folder)
    # print 'file_list: ', file_list
    no_of_frames = len(file_list)
    print 'no_of_frames: ', no_of_frames

    end_id = no_of_frames

    init_corners = np.asarray([ground_truth[0, 0:2].tolist(),
                               ground_truth[0, 2:4].tolist(),
                               ground_truth[0, 4:6].tolist(),
                               ground_truth[0, 6:8].tolist()]).T

    std_pts, std_corners = getNormalizedUnitSquarePts(std_resx, std_resy)
    std_pts_hm = util.homogenize(std_pts)
    std_corners_hm = util.homogenize(std_corners)
    (init_corners_norm, init_norm_mat) = getNormalizedPoints(init_corners)
    init_hom_mat = np.mat(util.compute_homography(std_corners, init_corners_norm))
    init_pts_norm = util.dehomogenize(init_hom_mat * std_pts_hm)
    init_pts = util.dehomogenize(init_norm_mat * util.homogenize(init_pts_norm))

    init_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(1))
    init_img_gs = cv2.cvtColor(init_img, cv2.cv.CV_BGR2GRAY).astype(np.float64)
    if filter_type != 'none':
        init_img_gs = applyFilter(init_img_gs, filter_type, kernel_size)
    init_img_gs = pre_proc_func(init_img_gs)

    init_pixel_vals = np.mat([util.bilin_interp(init_img_gs, init_pts[0, pt_id], init_pts[1, pt_id]) for pt_id in
                              xrange(n_pts)])
    init_pixel_vals = post_proc_func(init_pixel_vals)

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
        curr_corners_norm = init_corners_norm
        curr_norm_mat = init_norm_mat
        curr_hom_mat = init_hom_mat
        curr_pts = init_pts

    gt_col = (0, 0, 0)
    trans_col = (255, 255, 255)
    rs_col = (0, 0, 255)
    shear_col = (0, 255, 0)
    proj_col = (255, 255, 0)

    window_name = 'Permuted Piecewise Optimization'
    if show_img:
        cv2.namedWindow(window_name)

    track_img_dir = '{:s}/{:s}_{:s}'.format(track_img_root_dir, appearance_model, seq_name)
    track_data_dir = '{:s}/{:s}_{:s}'.format(track_root_dir, appearance_model, seq_name)
    dist_dir = '{:s}/{:s}_{:s}'.format(dist_root_dir, appearance_model, seq_name)

    if filter_type != 'none':
        track_img_dir = track_img_dir + '_' + filter_type + str(kernel_size)
        track_data_dir = track_data_dir + '_' + filter_type + str(kernel_size)
        dist_dir = dist_dir + '_' + filter_type + str(kernel_size)

    if write_img:
        if not os.path.exists(track_img_dir):
            os.makedirs(track_img_dir)

    tracker_type = 'ppw' + str(selective_opt) + '_' + opt_type + '_' + inc_type

    if not os.path.exists(dist_dir):
        if read_dist_data:
            raise IOError('Folder containing distance data does not exist: ' + dist_dir)
        elif write_dist_data:
            os.makedirs(dist_dir)

    if not os.path.exists(track_data_dir):
        os.makedirs(track_data_dir)

    tracking_corners_fname = '{:s}/{:s}_corners.txt'.format(track_data_dir, tracker_type)
    tracking_params_fname = '{:s}/{:s}_params_inv.txt'.format(track_data_dir, tracker_type)
    tracking_dist_fname = '{:s}/{:s}_dist.txt'.format(track_data_dir, tracker_type)

    tracking_corners_fid = open(tracking_corners_fname, 'w')
    tracking_params_fid = open(tracking_params_fname, 'w')
    tracking_dist_fid = open(tracking_dist_fname, 'w')

    tracking_corners_fid.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
    writeCorners(tracking_corners_fid, init_corners, 1)

    if opt_type == 'ind':
        trans_dist_fname = dist_dir + '/' + 'gt' + '_' + inc_type + '_trans.bin'
        rs_dist_fname = dist_dir + '/' + 'gt' + '_' + inc_type + '_rs.bin'
        shear_dist_fname = dist_dir + '/' + 'gt' + '_' + inc_type + '_shear.bin'
        proj_dist_fname = dist_dir + '/' + 'gt' + '_' + inc_type + '_proj.bin'
    else:
        trans_dist_fname = dist_dir + '/' + tracker_type + '_trans.bin'
        rs_dist_fname = dist_dir + '/' + tracker_type + '_rs.bin'
        shear_dist_fname = dist_dir + '/' + tracker_type + '_shear.bin'
        proj_dist_fname = dist_dir + '/' + tracker_type + '_proj.bin'

    # dist_fids=[]
    trans_dist_fid = rs_dist_fid = shear_dist_fid = proj_dist_fid = None
    trans_params = rs_params = shear_params = proj_params = None
    trans_grid_size = rs_grid_size = shear_grid_size = proj_grid_size = 0
    if read_dist_data:
        if not os.path.isfile(trans_dist_fname):
            raise IOError('Trans distance data file is not present: {:s}'.format(trans_dist_fname))
        print 'Opening file:', trans_dist_fname
        trans_dist_fid = open(trans_dist_fname, 'rb')
        trans_params = np.fromfile(trans_dist_fid, dtype=np.uint32, count=4)
        trans_grid_size = trans_params[2] * trans_params[3]
        ty_vec = np.fromfile(trans_dist_fid, dtype=np.float64, count=trans_params[2])
        tx_vec = np.fromfile(trans_dist_fid, dtype=np.float64, count=trans_params[3])

        # dist_fids.append(trans_dist_fid)

        if not os.path.isfile(rs_dist_fname):
            raise IOError('RS distance data file is not present: {:s}'.format(rs_dist_fname))
        print 'Opening file:', rs_dist_fname
        rs_dist_fid = open(rs_dist_fname, 'rb')
        rs_params = np.fromfile(rs_dist_fid, dtype=np.uint32, count=4)
        rs_grid_size = rs_params[2] * rs_params[3]
        theta_vec = np.fromfile(rs_dist_fid, dtype=np.float64, count=rs_params[2])
        scale_vec = np.fromfile(rs_dist_fid, dtype=np.float64, count=rs_params[3])

        # dist_fids.append(rs_dist_fid)

        if not os.path.isfile(shear_dist_fname):
            raise IOError('Shear distance data file is not present: {:s}'.format(shear_dist_fname))
        print 'Opening file:', shear_dist_fname
        shear_dist_fid = open(shear_dist_fname, 'rb')
        shear_params = np.fromfile(shear_dist_fid, dtype=np.uint32, count=4)
        shear_grid_size = shear_params[2] * shear_params[3]
        b_vec = np.fromfile(shear_dist_fid, dtype=np.float64, count=shear_params[2])
        a_vec = np.fromfile(shear_dist_fid, dtype=np.float64, count=shear_params[3])

        # dist_fids.append(shear_dist_fid)

        if not os.path.isfile(proj_dist_fname):
            raise IOError('Proj distance data file is not present: {:s}'.format(proj_dist_fname))
        print 'Opening file:', proj_dist_fname
        proj_dist_fid = open(proj_dist_fname, 'rb')
        proj_params = np.fromfile(proj_dist_fid, dtype=np.uint32, count=4)
        proj_grid_size = proj_params[2] * proj_params[3]
        v2_vec = np.fromfile(proj_dist_fid, dtype=np.float64, count=proj_params[2])
        v1_vec = np.fromfile(proj_dist_fid, dtype=np.float64, count=proj_params[3])

        # dist_fids.append(proj_dist_fid)

    elif write_dist_data:
        trans_dist_fid = open(trans_dist_fname, 'wb')
        np.array([start_id - 1, start_id, ty_vec.size, tx_vec.size], dtype=np.uint32).tofile(trans_dist_fid)
        ty_vec.tofile(trans_dist_fid)
        tx_vec.tofile(trans_dist_fid)

        rs_dist_fid = open(rs_dist_fname, 'wb')
        np.array([start_id - 1, start_id, theta_vec.size, scale_vec.size], dtype=np.uint32).tofile(rs_dist_fid)
        theta_vec.tofile(rs_dist_fid)
        scale_vec.tofile(rs_dist_fid)

        shear_dist_fid = open(shear_dist_fname, 'wb')
        np.array([start_id - 1, start_id, b_vec.size, a_vec.size], dtype=np.uint32).tofile(shear_dist_fid)
        b_vec.tofile(shear_dist_fid)
        a_vec.tofile(shear_dist_fid)

        proj_dist_fid = open(proj_dist_fname, 'wb')
        np.array([start_id - 1, start_id, v2_vec.size, v1_vec.size], dtype=np.uint32).tofile(proj_dist_fid)
        v2_vec.tofile(proj_dist_fid)
        v1_vec.tofile(proj_dist_fid)

    grid_types = [
        'trans',
        'rs',
        'shear',
        'proj'
    ]
    grid_vec_types = [
        ['tx', 'ty'],
        ['scale', 'theta'],
        ['a', 'b'],
        ['v1', 'v2']
    ]
    grid_functions = [
        getTransDistanceGridPre,
        getRSDistanceGridComp,
        getShearDistanceGridComp,
        getProjDistanceGridComp
    ]
    grid_vectors = [
        [tx_vec, ty_vec],
        [scale_vec, theta_vec],
        [a_vec, b_vec],
        [v1_vec, v2_vec]
    ]
    dist_fids = [
        trans_dist_fid,
        rs_dist_fid,
        shear_dist_fid,
        proj_dist_fid
    ]
    print 'dist_fids: ', dist_fids
    grid_sizes = [
        trans_grid_size,
        rs_grid_size,
        shear_grid_size,
        proj_grid_size
    ]
    grid_params_list = [
        trans_params,
        rs_params,
        shear_params,
        proj_params
    ]
    matrix_functions = [
        getTranslationMatrix,
        getRSMatrix,
        getShearingMatrix,
        getProjectionMatrix
    ]

    print 'theta_vec: ', theta_vec
    print 'scale_vec: ', scale_vec
    print 'a_vec: ', a_vec
    print 'b_vec: ', b_vec
    print 'v1_vec: ', v1_vec
    print 'v2_vec: ', v2_vec

    # print 'grid_vectors:\n', grid_vectors

    track_errors = []
    track_params = []
    gt_params = []

    no_of_grids = len(grid_types)

    if show_img:
        drawRegion(init_img, init_corners, gt_col, 1)
        cv2.imshow(window_name, init_img)

    end_exec = 0

    for frame_id in xrange(start_id, end_id):

        print '\n\nframe_id: ', frame_id, ' of ', end_id

        final_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id + 1))
        if final_img is None:
            break
        final_img_gs = cv2.cvtColor(final_img, cv2.cv.CV_BGR2GRAY).astype(np.float64)
        if filter_type != 'none':
            final_img_gs = applyFilter(final_img_gs, filter_type, kernel_size)

        final_img_gs = pre_proc_func(final_img_gs)

        interpolated_images = getLinearInterpolatedImages(init_img_gs, final_img_gs, n_interp)
        interpolated_images.append(final_img_gs)

        if use_gt:
            gt_corners = np.asarray([ground_truth[frame_id, 0:2].tolist(),
                                     ground_truth[frame_id, 2:4].tolist(),
                                     ground_truth[frame_id, 4:6].tolist(),
                                     ground_truth[frame_id, 6:8].tolist()]).T
            (gt_corners_norm, gt_norm_mat) = getNormalizedPoints(gt_corners)

            if show_gt:
                drawRegion(final_img, gt_corners, gt_col, 1)
                cv2.putText(final_img, "Actual", (int(gt_corners[0, 0]), int(gt_corners[1, 0])),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, gt_col)

        for curr_img_gs in interpolated_images:

            grid_permutations = itertools.permutations(range(no_of_grids))

            min_dist_prod = None
            min_dist_grids = min_opt_params = min_opt_dist = min_opt_corners = min_grid_ids = None

            start_time = time.clock()

            for curr_grid_ids in grid_permutations:
                curr_grid_types = []
                for grid_id in curr_grid_ids:
                    curr_grid_types.append(grid_types[grid_id])
                print '\tcurr_grid_ids: ', curr_grid_ids
                print '\tcurr_grid_types: ', curr_grid_types

                curr_dist_grids = [None] * no_of_grids
                curr_opt_params = [None] * no_of_grids
                curr_opt_dist = [None] * no_of_grids

                pw_opt_mat = np.identity(3)
                pw_hom_mat = curr_hom_mat

                dist_prod = 1

                for grid_id in curr_grid_ids:
                    x_vec, y_vec = grid_vectors[grid_id]
                    # print 'x_vec: ', x_vec
                    # print 'y_vec: ', y_vec

                    if read_dist_data:
                        dist_fid = dist_fids[grid_id]
                        grid_params = grid_params_list[grid_id]
                        grid_size = grid_sizes[grid_id]
                        dist_grid = np.fromfile(dist_fid, dtype=np.float64, count=grid_size).reshape(
                            [grid_params[2], grid_params[3]])
                    else:
                        grid_func = grid_functions[grid_id]
                        dist_grid = grid_func(std_pts, pw_hom_mat, curr_norm_mat, curr_img_gs,
                                              init_pixel_vals, x_vec, y_vec, dist_func)

                    row_id, col_id = opt_func(dist_grid)
                    x = x_vec[row_id]
                    y = y_vec[col_id]
                    dist = dist_grid[row_id, col_id]
                    dist_prod *= dist

                    mat_func = matrix_functions[grid_id]
                    opt_mat = mat_func(x, y)
                    pw_opt_mat = pw_opt_mat * opt_mat
                    pw_hom_mat = curr_hom_mat * pw_opt_mat

                    curr_dist_grids[grid_id] = np.copy(dist_grid)
                    curr_opt_params[grid_id] = [x, y]
                    curr_opt_dist[grid_id] = dist

                    grid_type = grid_types[grid_id]
                    # print '\t\t{:6s}: dist: {:f}'.format(grid_type, dist)
                    if show_img:
                        key = cv2.waitKey(1)
                        if key == 27:
                            end_exec = 1
                if end_exec:
                    break

                curr_opt_corners = util.dehomogenize(curr_norm_mat * pw_hom_mat * std_corners_hm)
                if min_dist_prod is None or is_better(dist_prod, min_dist_prod):
                    min_dist_prod = dist_prod
                    min_grid_ids = curr_grid_ids[:]
                    min_dist_grids = curr_dist_grids[:]
                    min_opt_params = curr_opt_params[:]
                    min_opt_dist = curr_opt_dist[:]
                    min_opt_corners = np.copy(curr_opt_corners)

                print '\tdist_prod: ', dist_prod
                print '\tmin_dist_prod: ', min_dist_prod
                # print 'curr_opt_dist: ', curr_opt_dist
                # break

            curr_corners = min_opt_corners
            # print 'Writing curr_corners:\n', curr_corners, '\nto file: \n', tracking_corners_fid
            writeCorners(tracking_corners_fid, curr_corners, frame_id + 1)
            curr_corners_norm, curr_norm_mat = getNormalizedPoints(curr_corners)
            try:
                curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners_norm))
            except:
                print 'Error: SVD did not converge while computing homography for: \n', curr_corners_norm
                break
            curr_pts_norm = util.dehomogenize(curr_hom_mat * std_pts_hm)
            curr_pts = util.dehomogenize(curr_norm_mat * util.homogenize(curr_pts_norm))

        if end_exec:
            break

        end_time = time.clock()
        curr_time = end_time - start_time
        # curr_fps = 1.0 / curr_time
        print 'curr_time: ', curr_time

        min_grid_types = []
        min_vec_types = []
        for grid_id in min_grid_ids:
            min_grid_types.append(grid_types[grid_id])
            min_vec_types.append(grid_vec_types[grid_id])

        print 'min_dist_prod: ', min_dist_prod
        print 'min_grid_ids: ', min_grid_ids
        print 'min_grid_types: ', min_grid_types

        for grid_id in min_grid_ids:
            grid_type = grid_types[grid_id]
            x_type, y_type = grid_vec_types[grid_id]
            min_x, min_y = min_opt_params[grid_id]
            dist = min_opt_dist[grid_id]

            print '{:6s}: {:6s}: {:10.7f} {:6s}: {:10.7f} dist: {:12.6f}'.format(grid_type, x_type, min_x, y_type,
                                                                                 min_y, dist)
            if write_dist_data:
                dist_grid = min_dist_grids[grid_id]
                dist_fid = dist_fids[grid_id]
                # print 'Writing dist_grid to: ', dist_fid
                dist_grid.tofile(dist_fid)
                current_offset = dist_fid.tell()
                dist_fid.seek(0)
                np.array([frame_id], dtype=np.uint32).tofile(dist_fid)
                dist_fid.seek(current_offset)

        frame_params_txt = '{:4d}'.format(frame_id)
        frame_dist_txt = '{:4d}'.format(frame_id)
        for i in xrange(no_of_grids):
            grid_type = grid_types[i]
            min_x, min_y = min_opt_params[i]
            frame_dist_txt += '\t{:s}\t{:12.8f}'.format(min_grid_types[i], min_opt_dist[i])
            if grid_type == 'rs':
                frame_params_txt += '\t{:12.8f}\t{:12.8f}'.format(min_y, min_x)
            else:
                frame_params_txt += '\t{:12.8f}\t{:12.8f}'.format(min_x, min_y)
        frame_params_txt += '\n'
        frame_dist_txt += '\n'
        # print 'Writing frame_params_txt:\n', frame_params_txt, '\nto file: \n', tracking_params_fid
        tracking_params_fid.write(frame_params_txt)
        # print 'Writing frame_dist_txt:\n', frame_dist_txt, '\nto file: \n', tracking_dist_fid
        tracking_dist_fid.write(frame_dist_txt)

        curr_corners = min_opt_corners
        # print 'Writing curr_corners:\n', curr_corners, '\nto file: \n', tracking_corners_fid
        writeCorners(tracking_corners_fid, curr_corners, frame_id + 1)
        curr_corners_norm, curr_norm_mat = getNormalizedPoints(curr_corners)
        try:
            curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners_norm))
        except:
            print 'Error: SVD did not converge while computing homography for: \n', curr_corners_norm
            break
        curr_pts_norm = util.dehomogenize(curr_hom_mat * std_pts_hm)
        curr_pts = util.dehomogenize(curr_norm_mat * util.homogenize(curr_pts_norm))

        if show_img:
            drawRegion(final_img, curr_corners, proj_col, 1)
            cv2.imshow(window_name, final_img)
            key = cv2.waitKey(1)
            if key == 27:
                break

        init_img_gs = final_img_gs.copy()

    for dist_fid in dist_fids:
        dist_fid.close()

    tracking_corners_fid.close()
    tracking_params_fid.close()
    tracking_dist_fid.close()


