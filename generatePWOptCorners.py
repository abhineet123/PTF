from distanceGrid import *
import time
import os
from Misc import *

if __name__ == '__main__':

    db_root_dir = '../Datasets'
    track_root_dir = '../Tracking Data'
    img_root_dir = '../Image Data'
    dist_root_dir = '../Distance Data'
    track_img_root_dir = '../Tracked Images'

    actor = 'Human'

    params_dict = getParamDict()
    param_ids = readDistGridParams()

    actors = params_dict['actors']
    sequences = params_dict['sequences']
    grid_types = params_dict['grid_types']
    inc_types = params_dict['inc_types']
    appearance_models = params_dict['appearance_models']
    tracker_types = params_dict['tracker_types']
    filter_types = params_dict['filter_types']
    opt_types = params_dict['opt_types']
    challenges = params_dict['challenges']

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
    upd_template = param_ids['upd_template']

    n_bins = param_ids['n_bins']
    dof = param_ids['dof']
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
    print 'dof: ', dof
    print 'read_dist_data: ', read_dist_data
    print 'write_dist_data: ', write_dist_data
    print 'upd_template: ', upd_template

    dist_func, pre_proc_func, post_proc_func, opt_func, is_better = getDistanceFunction(appearance_model, n_pts, n_bins)

    src_folder = db_root_dir + '/' + actor + '/' + seq_name

    ground_truth_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)

    no_of_frames = ground_truth.shape[0]
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

    window_name = 'Piecewise Optimization'
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

    tracker_type = 'pw'
    if upd_template:
        tracker_type += 'u'
    tracker_type += str(selective_opt) + '_' + opt_type + str(dof) + '_' + inc_type
    print 'tracker_type: ', tracker_type

    if not os.path.exists(dist_dir):
        if read_dist_data:
            raise IOError('Folder containing distance data does not exist: ' + dist_dir)
        elif write_dist_data:
            os.makedirs(dist_dir)

    if not os.path.exists(track_data_dir):
        os.makedirs(track_data_dir)

    tracking_corners_fname = '{:s}/{:s}_corners.txt'.format(track_data_dir, tracker_type)
    tracking_errors_fname = '{:s}/{:s}_errors.txt'.format(track_data_dir, tracker_type)
    tracking_params_fname = '{:s}/{:s}_params_inv.txt'.format(track_data_dir, tracker_type)

    tracking_corners_fid = open(tracking_corners_fname, 'w')
    tracking_corners_fid.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
    writeCorners(tracking_corners_fid, init_corners, 1)

    tracking_errors_fid = open(tracking_errors_fname, 'w')
    tracking_params_fid = open(tracking_params_fname, 'w')

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

    if read_dist_data:
        if dof >= 2:
            if not os.path.isfile(trans_dist_fname):
                raise IOError('Trans distance data file is not present: {:s}'.format(trans_dist_fname))
            print 'Opening file:', trans_dist_fname
            trans_dist_fid = open(trans_dist_fname, 'rb')
            trans_params = np.fromfile(trans_dist_fid, dtype=np.uint32, count=4)
            trans_grid_size = trans_params[2] * trans_params[3]
            ty_vec = np.fromfile(trans_dist_fid, dtype=np.float64, count=trans_params[2])
            tx_vec = np.fromfile(trans_dist_fid, dtype=np.float64, count=trans_params[3])

        if dof >= 4:
            if not os.path.isfile(rs_dist_fname):
                raise IOError('RS distance data file is not present: {:s}'.format(rs_dist_fname))
            print 'Opening file:', rs_dist_fname
            rs_dist_fid = open(rs_dist_fname, 'rb')
            rs_params = np.fromfile(rs_dist_fid, dtype=np.uint32, count=4)
            rs_grid_size = rs_params[2] * rs_params[3]
            theta_vec = np.fromfile(rs_dist_fid, dtype=np.float64, count=rs_params[2])
            scale_vec = np.fromfile(rs_dist_fid, dtype=np.float64, count=rs_params[3])

        if dof >= 6:
            if not os.path.isfile(shear_dist_fname):
                raise IOError('Shear distance data file is not present: {:s}'.format(shear_dist_fname))
            print 'Opening file:', shear_dist_fname
            shear_dist_fid = open(shear_dist_fname, 'rb')
            shear_params = np.fromfile(shear_dist_fid, dtype=np.uint32, count=4)
            shear_grid_size = shear_params[2] * shear_params[3]
            b_vec = np.fromfile(shear_dist_fid, dtype=np.float64, count=shear_params[2])
            a_vec = np.fromfile(shear_dist_fid, dtype=np.float64, count=shear_params[3])

        if dof >= 8:
            if not os.path.isfile(proj_dist_fname):
                raise IOError('Proj distance data file is not present: {:s}'.format(proj_dist_fname))
            print 'Opening file:', proj_dist_fname
            proj_dist_fid = open(proj_dist_fname, 'rb')
            proj_params = np.fromfile(proj_dist_fid, dtype=np.uint32, count=4)
            proj_grid_size = proj_params[2] * proj_params[3]
            v2_vec = np.fromfile(proj_dist_fid, dtype=np.float64, count=proj_params[2])
            v1_vec = np.fromfile(proj_dist_fid, dtype=np.float64, count=proj_params[3])

    elif write_dist_data:
        if dof >= 2:
            trans_dist_fid = open(trans_dist_fname, 'wb')
            np.array([start_id - 1, start_id, ty_vec.size, tx_vec.size], dtype=np.uint32).tofile(trans_dist_fid)
            ty_vec.tofile(trans_dist_fid)
            tx_vec.tofile(trans_dist_fid)

        if dof >= 4:
            rs_dist_fid = open(rs_dist_fname, 'wb')
            np.array([start_id - 1, start_id, theta_vec.size, scale_vec.size], dtype=np.uint32).tofile(rs_dist_fid)
            theta_vec.tofile(rs_dist_fid)
            scale_vec.tofile(rs_dist_fid)

        if dof >= 6:
            shear_dist_fid = open(shear_dist_fname, 'wb')
            np.array([start_id - 1, start_id, b_vec.size, a_vec.size], dtype=np.uint32).tofile(shear_dist_fid)
            b_vec.tofile(shear_dist_fid)
            a_vec.tofile(shear_dist_fid)

        if dof >= 8:
            proj_dist_fid = open(proj_dist_fname, 'wb')
            np.array([start_id - 1, start_id, v2_vec.size, v1_vec.size], dtype=np.uint32).tofile(proj_dist_fid)
            v2_vec.tofile(proj_dist_fid)
            v1_vec.tofile(proj_dist_fid)

    print 'theta_vec: ', theta_vec
    print 'scale_vec: ', scale_vec
    print 'a_vec: ', a_vec
    print 'b_vec: ', b_vec
    print 'v1_vec: ', v1_vec
    print 'v2_vec: ', v2_vec

    track_errors = []
    track_params = []
    gt_params = []

    for frame_id in xrange(start_id, end_id):

        print 'frame_id: ', frame_id
        err_text = ''

        frame_err_txt = '{:4d}'.format(frame_id)
        frame_params_txt = '{:4d}'.format(frame_id)
        frame_err = [frame_id + 1]
        frame_params = [frame_id + 1]
        pw_opt_mat = np.identity(3)

        curr_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id + 1))
        if curr_img is None:
            break
        curr_img_gs = cv2.cvtColor(curr_img, cv2.cv.CV_BGR2GRAY).astype(np.float64)
        if filter_type != 'none':
            curr_img_gs = applyFilter(curr_img_gs, filter_type, kernel_size)

        curr_img_gs = pre_proc_func(curr_img_gs)

        gt_corners = np.asarray([ground_truth[frame_id, 0:2].tolist(),
                                 ground_truth[frame_id, 2:4].tolist(),
                                 ground_truth[frame_id, 4:6].tolist(),
                                 ground_truth[frame_id, 6:8].tolist()]).T
        (gt_corners_norm, gt_norm_mat) = getNormalizedPoints(gt_corners)

        if show_gt:
            drawRegion(curr_img, gt_corners, gt_col, 1)
            cv2.putText(curr_img, "Actual", (int(gt_corners[0, 0]), int(gt_corners[1, 0])),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, gt_col)

        start_time = time.clock()

        if inc_type == 'fc':
            if dof >= 2:
                # ------------------------------------ TRANSLATION ------------------------------------#
                if read_dist_data:
                    trans_dist_grid = np.fromfile(trans_dist_fid, dtype=np.float64, count=trans_grid_size).reshape(
                        [trans_params[2], trans_params[3]])
                else:
                    trans_dist_grid = getTransDistanceGridPre(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs,
                                                              init_pixel_vals, tx_vec, ty_vec, dist_func)
                if write_dist_data:
                    trans_dist_grid.tofile(trans_dist_fid)
                    current_offset = trans_dist_fid.tell()
                    trans_dist_fid.seek(0)
                    np.array([frame_id], dtype=np.uint32).tofile(trans_dist_fid)
                    trans_dist_fid.seek(current_offset)

                row_id, col_id = opt_func(trans_dist_grid)
                tx = tx_vec[row_id]
                ty = ty_vec[col_id]
                if frame_id > start_id and selective_opt:
                    if row_id < selective_thresh or \
                                    row_id > tx_vec.size - selective_thresh or \
                                    col_id < selective_thresh or \
                                    col_id > ty_vec.size - selective_thresh:
                        tx, ty = track_params[-1][1:3]

                trans_opt_mat = getTranslationMatrix(tx, ty)
                frame_params.append(tx)
                frame_params.append(ty)

                pw_opt_mat = trans_opt_mat
                trans_hom_mat = curr_hom_mat * pw_opt_mat
                trans_opt_corners = util.dehomogenize(curr_norm_mat * trans_hom_mat * std_corners_hm)
                curr_corners = trans_opt_corners

                trans_error = math.sqrt(np.sum(np.square(trans_opt_corners - gt_corners)) / 4)
                frame_err.append(trans_error)

                frame_params_txt += '\t{:12.8f}\t{:12.8f}'.format(tx, ty)
                frame_err_txt += '\t{:12.8f}'.format(trans_error)
                err_text = err_text + 'trans: {:7.4f} '.format(trans_error)

                # writeCorners(trans_corners_fid, curr_corners, frame_id + 1)

                print 'trans:: tx: ', tx, 'ty: ', ty, 'dist:', trans_dist_grid[
                    row_id, col_id], 'error:', trans_error
                if show_trans:
                    drawRegion(curr_img, trans_opt_corners, trans_col, 1)
                    cv2.putText(curr_img, "Trans", (int(trans_opt_corners[0, 0]), int(trans_opt_corners[1, 0])),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, trans_col)
            if dof >= 4:
                # ------------------------------------ ROTATION-SCALING ------------------------------------#

                if read_dist_data:
                    rs_dist_grid = np.fromfile(rs_dist_fid, dtype=np.float64, count=rs_grid_size).reshape(
                        [rs_params[2], rs_params[3]])
                else:
                    if opt_type == 'pre':
                        pw_std_pts = util.dehomogenize(pw_opt_mat * std_pts_hm)
                        rs_dist_grid = getRSDistanceGridComp(pw_std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs,
                                                             init_pixel_vals,
                                                             scale_vec, theta_vec, dist_func)
                    elif opt_type == 'post':
                        pw_hom_mat = curr_hom_mat * pw_opt_mat
                        rs_dist_grid = getRSDistanceGridComp(std_pts, pw_hom_mat, curr_norm_mat, curr_img_gs,
                                                             init_pixel_vals,
                                                             scale_vec, theta_vec, dist_func)
                    elif opt_type == 'ind':
                        rs_dist_grid = getRSDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs,
                                                             init_pixel_vals,
                                                             scale_vec, theta_vec, dist_func)

                if write_dist_data:
                    rs_dist_grid.tofile(rs_dist_fid)
                    current_offset = rs_dist_fid.tell()
                    rs_dist_fid.seek(0)
                    np.array([frame_id], dtype=np.uint32).tofile(rs_dist_fid)
                    rs_dist_fid.seek(current_offset)

                row_id, col_id = opt_func(rs_dist_grid)
                scale = scale_vec[row_id]
                theta = theta_vec[col_id]

                if frame_id > start_id and selective_opt:
                    if row_id < selective_thresh or \
                                    row_id > scale_vec.size - selective_thresh or \
                                    col_id < selective_thresh or \
                                    col_id > theta_vec.size - selective_thresh:
                        theta, scale = track_params[-1][3:5]

                rot_opt_mat = getRotationMatrix(theta)
                scale_opt_mat = getScalingMatrix(scale)
                frame_params.append(theta)
                frame_params.append(scale)

                if opt_type == 'pre' or opt_type == 'ind':
                    pw_opt_mat = scale_opt_mat * rot_opt_mat * pw_opt_mat
                elif opt_type == 'post':
                    pw_opt_mat = pw_opt_mat * scale_opt_mat * rot_opt_mat

                rs_hom_mat = curr_hom_mat * pw_opt_mat
                rs_opt_corners = util.dehomogenize(curr_norm_mat * rs_hom_mat * std_corners_hm)
                curr_corners = rs_opt_corners

                rs_error = math.sqrt(np.sum(np.square(rs_opt_corners - gt_corners)) / 4)
                frame_err.append(rs_error)

                frame_params_txt += '\t{:12.8f}\t{:12.8f}'.format(theta, scale)
                frame_err_txt += '\t{:12.8f}'.format(rs_error)
                err_text = err_text + 'rs: {:7.4f} '.format(rs_error)
                print 'rs:: theta: ', theta, 'scale: ', scale, 'dist:', rs_dist_grid[row_id, col_id], 'error:', rs_error

                # writeCorners(rs_corners_fid, curr_corners, frame_id + 1)
                if show_rs:
                    drawRegion(curr_img, rs_opt_corners, rs_col, 1)
                    cv2.putText(curr_img, "RS", (int(rs_opt_corners[0, 0]), int(rs_opt_corners[1, 0])),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, rs_col)
                    # cv2.imshow(window_name, curr_img)
                    # cv2.waitKey(0)

            if dof >= 6:
                # ------------------------------------ SHEAR ------------------------------------#

                if read_dist_data:
                    shear_dist_grid = np.fromfile(shear_dist_fid, dtype=np.float64, count=shear_grid_size).reshape(
                        [shear_params[2], shear_params[3]])
                else:
                    if opt_type == 'pre':
                        pw_std_pts = util.dehomogenize(pw_opt_mat * std_pts_hm)
                        shear_dist_grid = getShearDistanceGridComp(pw_std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs,
                                                                   init_pixel_vals,
                                                                   a_vec, b_vec, dist_func)
                    elif opt_type == 'post':
                        pw_hom_mat = curr_hom_mat * pw_opt_mat
                        shear_dist_grid = getShearDistanceGridComp(std_pts, pw_hom_mat, curr_norm_mat, curr_img_gs,
                                                                   init_pixel_vals,
                                                                   a_vec, b_vec, dist_func)
                    elif opt_type == 'ind':
                        shear_dist_grid = getShearDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs,
                                                                   init_pixel_vals,
                                                                   a_vec, b_vec, dist_func)

                if write_dist_data:
                    shear_dist_grid.tofile(shear_dist_fid)
                    current_offset = shear_dist_fid.tell()
                    shear_dist_fid.seek(0)
                    np.array([frame_id], dtype=np.uint32).tofile(shear_dist_fid)
                    shear_dist_fid.seek(current_offset)

                row_id, col_id = opt_func(shear_dist_grid)
                a = a_vec[row_id]
                b = b_vec[col_id]
                if frame_id > start_id and selective_opt:
                    if row_id < selective_thresh or \
                                    row_id > a_vec.size - selective_thresh or \
                                    col_id < selective_thresh or \
                                    col_id > b_vec.size - selective_thresh:
                        a, b = track_params[-1][5:7]

                shear_opt_mat = getShearingMatrix(a, b)
                frame_params.append(a)
                frame_params.append(b)

                if opt_type == 'pre' or opt_type == 'ind':
                    pw_opt_mat = shear_opt_mat * pw_opt_mat
                elif opt_type == 'post':
                    pw_opt_mat = pw_opt_mat * shear_opt_mat

                shear_hom_mat = curr_hom_mat * pw_opt_mat
                shear_opt_corners = util.dehomogenize(curr_norm_mat * shear_hom_mat * std_corners_hm)
                curr_corners = shear_opt_corners

                shear_error = math.sqrt(np.sum(np.square(shear_opt_corners - gt_corners)) / 4)
                frame_err.append(shear_error)

                frame_params_txt += '\t{:12.8f}\t{:12.8f}'.format(a, b)
                frame_err_txt += '\t{:12.8f}'.format(shear_error)
                err_text = err_text + 'shear: {:7.4f} '.format(shear_error)

                # writeCorners(shear_corners_fid, curr_corners, frame_id + 1)

                print 'shear:: a: ', a, 'b: ', b, 'dist:', shear_dist_grid[
                    row_id, col_id], 'error:', shear_error
                if show_shear:
                    drawRegion(curr_img, shear_opt_corners, shear_col, 1)
                    cv2.putText(curr_img, "Shear", (int(shear_opt_corners[0, 0]), int(shear_opt_corners[1, 0])),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, shear_col)
            if dof == 8:
                # ------------------------------------ PROJECTION ------------------------------------#

                if read_dist_data:
                    proj_dist_grid = np.fromfile(proj_dist_fid, dtype=np.float64, count=proj_grid_size).reshape(
                        [proj_params[2], proj_params[3]])
                else:
                    if opt_type == 'pre':
                        pw_std_pts = util.dehomogenize(pw_opt_mat * std_pts_hm)
                        proj_dist_grid = getProjDistanceGridComp(pw_std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs,
                                                                 init_pixel_vals,
                                                                 v1_vec, v2_vec, dist_func)
                    elif opt_type == 'post':
                        pw_hom_mat = curr_hom_mat * pw_opt_mat
                        proj_dist_grid = getProjDistanceGridComp(std_pts, pw_hom_mat, curr_norm_mat, curr_img_gs,
                                                                 init_pixel_vals,
                                                                 v1_vec, v2_vec, dist_func)
                    elif opt_type == 'ind':
                        proj_dist_grid = getProjDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, curr_img_gs,
                                                                 init_pixel_vals,
                                                                 v1_vec, v2_vec, dist_func)
                if write_dist_data:
                    proj_dist_grid.tofile(proj_dist_fid)
                    current_offset = proj_dist_fid.tell()
                    proj_dist_fid.seek(0)
                    np.array([frame_id], dtype=np.uint32).tofile(proj_dist_fid)
                    proj_dist_fid.seek(current_offset)

                row_id, col_id = opt_func(proj_dist_grid)
                v1 = v1_vec[row_id]
                v2 = v2_vec[col_id]
                if frame_id > start_id and selective_opt:
                    if row_id < selective_thresh or \
                                    row_id > v1_vec.size - selective_thresh or \
                                    col_id < selective_thresh or \
                                    col_id > v2_vec.size - selective_thresh:
                        v1, v2 = track_params[-1][7:9]

                proj_opt_mat = getProjectionMatrix(v1, v2)
                frame_params.append(v1)
                frame_params.append(v2)

                # proj_std_pts_hm = proj_opt_mat * util.homogenize(pw_std_pts)
                # proj_std_pts = util.dehomogenize(proj_std_pts_hm)
                # proj_std_corners_hm = proj_opt_mat * pw_std_corners_hm

                if opt_type == 'pre' or opt_type == 'ind':
                    pw_opt_mat = proj_opt_mat * pw_opt_mat
                elif opt_type == 'post':
                    pw_opt_mat = pw_opt_mat * proj_opt_mat

                proj_hom_mat = curr_hom_mat * pw_opt_mat
                proj_opt_corners = util.dehomogenize(curr_norm_mat * proj_hom_mat * std_corners_hm)
                curr_corners = proj_opt_corners

                proj_error = math.sqrt(np.sum(np.square(proj_opt_corners - gt_corners)) / 4)
                frame_err.append(proj_error)

                frame_params_txt += '\t{:12.8f}\t{:12.8f}'.format(v1, v2)
                frame_err_txt += '\t{:12.8f}'.format(proj_error)
                err_text = err_text + 'proj: {:7.4f} '.format(proj_error)

                # writeCorners(tracking_corners_fid, curr_corners, frame_id + 1)

                print 'proj:: v1: ', v1, 'v2: ', v2, 'dist:', proj_dist_grid[row_id, col_id], 'error:', proj_error
                # print 'proj_opt_corners: ', proj_opt_corners
                if show_proj:
                    drawRegion(curr_img, proj_opt_corners, proj_col, 1)
                    cv2.putText(curr_img, "Proj", (int(proj_opt_corners[0, 0]), int(proj_opt_corners[1, 0])),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, proj_col)
            cv2.putText(curr_img, err_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

            if write_track_data:
                frame_err_txt += '\n'
                frame_params_txt += '\n'
                tracking_errors_fid.write(frame_err_txt)
                tracking_params_fid.write(frame_params_txt)
                track_errors.append(frame_err)
                track_params.append(frame_params)
        else:
            curr_corners = gt_corners

        end_time = time.clock()
        curr_time = end_time - start_time
        curr_fps = 1.0 / curr_time
        # print 'curr_time: ', curr_time, 'secs'
        writeCorners(tracking_corners_fid, curr_corners, frame_id + 1)

        if write_img:
            fname = '{:s}/frame{:05d}.jpg'.format(track_img_dir, frame_id)
            cv2.imwrite(fname, curr_img)

        if show_img:
            cv2.imshow(window_name, curr_img)
            key = cv2.waitKey(1)
            if key == 27:
                break

        # curr_corners = gt_corners
        curr_corners_norm, curr_norm_mat = getNormalizedPoints(curr_corners)
        try:
            curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners_norm))
        except:
            print 'Error: SVD did not converge while computing homography for: \n', curr_corners_norm
            break
        curr_pts_norm = util.dehomogenize(curr_hom_mat * std_pts_hm)
        curr_pts = util.dehomogenize(curr_norm_mat * util.homogenize(curr_pts_norm))
        if upd_template:
            init_pixel_vals = np.mat([util.bilin_interp(curr_img_gs, curr_pts[0, pt_id], curr_pts[1, pt_id]) for pt_id in
                                      xrange(n_pts)])
            init_pixel_vals = post_proc_func(init_pixel_vals)

    tracking_corners_fid.close()
    tracking_params_fid.close()
    tracking_errors_fid.close()

    if dof >= 2:
        trans_dist_fid.close()
    if dof >= 4:
        rs_dist_fid.close()
    if dof >= 6:
        shear_dist_fid.close()
    if dof >= 8:
        proj_dist_fid.close()

    print 'Exiting....'

