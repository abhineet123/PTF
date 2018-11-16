from optimization import *

if __name__ == '__main__':

    params_dict = getParamDict()
    param_ids = readDistGridParams()
    pause_seq = 0
    gt_col = (0, 255, 0)
    actors = params_dict['actors']
    sequences = params_dict['sequences']
    opt_methods = params_dict['opt_methods']

    actor_id = param_ids['actor_id']
    seq_id = param_ids['seq_id']
    opt_id = param_ids['opt_id']

    db_root_dir = '../Datasets'
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d.jpg'

    font_size = 0.8

    update_init_corners = 0
    print_corners = 0
    write_corners = 1

    # flags
    use_homography_ls = 0
    use_affine_ls = 0
    use_similarity_ls = 0
    perform_pw_opt = 0
    perform_sim_opt = 0
    initialize_with_rect = 0

    show_img = 0

    show_actual = 1
    show_hom = 0
    show_affine = 0
    show_sim = 0
    show_rt_opt = 0
    show_trs = 1
    show_trs_opt = 0
    show_trans = 0

    show_se2 = 0
    show_rt = 0
    show_trans_mean = 0
    show_rt_mean = 0
    show_se2_mean = 0
    show_rt_pwopt = 0
    show_se2_opt = 0
    show_se2_pwopt = 0

    arg_id = 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        opt_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        perform_pw_opt = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        perform_sim_opt = int(sys.argv[arg_id])
        arg_id += 1

    actor = actors[actor_id]
    sequences = sequences[actor]
    seq_name = sequences[seq_id]
    opt_method = opt_methods[opt_id]

    if seq_id >= len(sequences):
        print 'Invalid dataset_id: ', seq_id
        sys.exit()

    affine_jacobian_func = None
    se2_jacobian_func = None
    trs_jacobian_func = None
    rt_jacobian_func = None
    shear_jacobian_func = None
    rot_jacobian_func = None
    scale_jacobian_func = None
    trans_jacobian_func = None
    proj_jacobian_func = None

    conditioning_func = getNormalizedPoints

    if opt_id == 0 or opt_id == 5 or opt_id == 6:
        affine_jacobian_func = getJacobianAffine
        se2_jacobian_func = getJacobianSE2
        rt_jacobian_func = getJacobianRT
        shear_jacobian_func = getJacobianShear
        rot_jacobian_func = getJacobianRotate
        scale_jacobian_func = getJacobianScale
        trans_jacobian_func = getJacobianTrans

    src_fname = db_root_dir + '/' + actor + '/' + seq_name + '/' + img_name_fmt
    ground_truth_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)

    print 'actor: ', actor
    print 'seq_id:', seq_id, 'seq_name:', seq_name
    print 'opt_id:', opt_id, 'opt_method:', opt_method
    print 'write_corners:', write_corners
    print 'perform_pw_opt:', perform_pw_opt
    print 'perform_sim_opt:', perform_sim_opt

    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        print 'The video file ', src_fname, ' could not be opened'
        sys.exit()

    no_of_frames = ground_truth.shape[0]
    print 'no_of_frames: ', no_of_frames

    init_corners = np.asarray([ground_truth[0, 0:2].tolist(),
                               ground_truth[0, 2:4].tolist(),
                               ground_truth[0, 4:6].tolist(),
                               ground_truth[0, 6:8].tolist()]).T

    if initialize_with_rect:
        init_corners, init_corners2 = getRectangularApproximation(init_corners)

    # (init_corners_norm, base_norm_mat_inv) = shiftGeometricCenterToOrigin(init_corners)
    (init_corners_norm, base_norm_mat_inv) = conditioning_func(init_corners)
    init_corners_hm = np.mat(util.homogenize(init_corners))
    init_corners_norm_hm = np.mat(util.homogenize(init_corners_norm))
    # print 'init_corners:\n', init_corners
    # print 'init_corners2:\n', init_corners2

    if (show_img):
        ret, init_img = cap.read()
        window_name = 'Optimized GT'
        cv2.namedWindow(window_name)

    act_col = (0, 0, 0)
    hom_col = (0, 0, 255)
    affine_col = (0, 255, 0)
    sim_col = (255, 255, 0)
    trs_col = (255, 0, 0)
    trs_opt_col = (128, 0, 0)
    trans_col = (255, 255, 255)
    se2_col = (255, 255, 0)
    rt_col = (255, 0, 0)

    hom_errors = []
    affine_errors = []
    sim_errors = []
    trs_errors = []
    trans_mean_errors = []

    rt_mean_errors = []
    se2_mean_errors = []
    trans_opt_errors = []
    affine_opt_errors = []
    se2_opt_errors = []
    trs_opt_errors = []
    se2_trans_errors = []
    rt_opt_errors = []
    rt_trans_errors = []
    rt_pwopt_errors = []
    se2_pwopt_errors = []
    affine_pwopt_errors = []
    hom_pwopt_errors = []

    error_fps = []
    pw_error_fps = []

    hom_error = 0
    affine_error = 0
    trans_mean_error = 0
    trs_opt_error = 0
    rt_mean_error = 0
    rt_trans_error = 0
    se2_mean_error = 0
    trans_opt_error = 0
    affine_opt_error = 0
    se2_opt_error = 0
    se2_trans_error = 0
    rt_opt_error = 0
    rt_pwopt_error = 0
    se2_pwopt_error = 0
    affine_pwopt_error = 0
    hom_pwopt_error = 0

    curr_fps = 0
    pw_curr_fps = 0

    if write_corners:
        opt_gt_root_dir=db_root_dir + '/' + actor + '/OptGT/'
        if not os.path.exists(opt_gt_root_dir):
            os.makedirs(opt_gt_root_dir)
            
        hom_fname = opt_gt_root_dir + seq_name + '_8.txt'
        hom_file = open(hom_fname, 'w')
        writeCorners(hom_file, init_corners, 1, 1)

        affine_fname = opt_gt_root_dir + seq_name + '_6.txt'
        affine_file = open(affine_fname, 'w')
        writeCorners(affine_file, init_corners, 1, 1)

        sim_fname = opt_gt_root_dir + seq_name + '_4.txt'
        sim_file = open(sim_fname, 'w')
        writeCorners(sim_file, init_corners, 1, 1)

        trs_fname = opt_gt_root_dir + seq_name + '_3s.txt'
        trs_file = open(trs_fname, 'w')
        writeCorners(trs_file, init_corners, 1, 1)

        rt_opt_fname = opt_gt_root_dir + seq_name + '_3.txt'
        rt_opt_file = open(rt_opt_fname, 'w')
        writeCorners(rt_opt_file, init_corners, 1, 1)

        if initialize_with_rect:
            trans_fname = opt_gt_root_dir + seq_name + '_2r.txt'
        else:
            trans_fname = opt_gt_root_dir + seq_name + '_2.txt'
        trans_file = open(trans_fname, 'w')
        writeCorners(trans_file, init_corners, 1, 1)

        if perform_pw_opt:
            rt_pwopt_fname = db_root_dir + '/' + actor + '/' + seq_name + '_rt_pwopt.txt'
            rt_pwopt_file = open(rt_pwopt_fname, 'w')
            writeCorners(rt_pwopt_file, init_corners)
            se2_pwopt_fname = db_root_dir + '/' + actor + '/' + seq_name + '_se2_pwopt.txt'
            se2_pwopt_file = open(se2_pwopt_fname, 'w')
            writeCorners(se2_pwopt_file, init_corners)

        if perform_sim_opt:
            # rt_mean_fname = db_root_dir + '/' + actor + '/' + seq_name + '_rt_mean.txt'
            # rt_mean_file = open(rt_mean_fname, 'w')
            # writeCorners(rt_mean_file, init_corners)
            #
            # se2_mean_fname = db_root_dir + '/' + actor + '/' + seq_name + '_se2_mean.txt'
            # se2_mean_file = open(se2_mean_fname, 'w')
            # writeCorners(se2_mean_file, init_corners)

            se2_opt_fname = db_root_dir + '/' + actor + '/' + seq_name + '_4.txt'
            se2_opt_file = open(se2_opt_fname, 'w')
            writeCorners(se2_opt_file, init_corners)

    states = ['off', 'on']

    for i in xrange(1, no_of_frames):

        curr_corners = np.asarray([ground_truth[i, 0:2].tolist(),
                                   ground_truth[i, 2:4].tolist(),
                                   ground_truth[i, 4:6].tolist(),
                                   ground_truth[i, 6:8].tolist()]).T
        (curr_corners_norm, curr_norm_mat_inv) = conditioning_func(curr_corners)
        curr_corners_hm = util.homogenize(curr_corners)
        curr_corners_norm_hm = util.homogenize(curr_corners_norm)
        # print 'curr_corners:\n', curr_corners
        # print 'curr_corners_norm:\n', curr_corners_norm

        if use_homography_ls:
            hom_mat = computeHomographyLS(init_corners_norm, curr_corners_norm)
        else:
            hom_mat = np.mat(util.compute_homography(init_corners_norm, curr_corners_norm))

        hom_corners = util.dehomogenize(curr_norm_mat_inv * hom_mat * init_corners_norm_hm)
        hom_error = math.sqrt(np.sum(np.square(hom_corners - curr_corners)) / 4)
        hom_errors.append(hom_error)

        affine_mat = computeAffineLS(init_corners, curr_corners)
        affine_corners = util.dehomogenize(affine_mat * init_corners_hm)
        affine_error = math.sqrt(np.sum(np.square(affine_corners - curr_corners)) / 4)
        affine_errors.append(affine_error)

        sim_mat = computeSimilarityLS(init_corners, curr_corners)
        sim_corners = util.dehomogenize(sim_mat * init_corners_hm)
        sim_error = math.sqrt(np.sum(np.square(sim_corners - curr_corners)) / 4)
        sim_errors.append(sim_error)

        trs_mat = computeTranscalingyLS(init_corners, curr_corners)
        trs_corners = util.dehomogenize(trs_mat * init_corners_hm)
        trs_error = math.sqrt(np.sum(np.square(trs_corners - curr_corners)) / 4)
        trs_errors.append(trs_error)

        rt_opt_mat, rt_trans_mat, rt_rot_mat = computeRTOpt(init_corners_hm, curr_corners_hm, opt_method,
                                                            jacobian_func=rt_jacobian_func)
        rt_opt_corners = util.dehomogenize(rt_opt_mat * init_corners_hm)
        rt_opt_error = math.sqrt(np.sum(np.square(rt_opt_corners - curr_corners)) / 4)
        rt_opt_errors.append(rt_opt_error)

        # trs_opt_mat = computeTranscalingOpt(init_corners_hm, curr_corners_hm,
        # opt_method, jacobian_func=trs_jacobian_func)
        # trs_opt_corners = util.dehomogenize(trs_opt_mat * init_corners_hm)
        # trs_opt_error = math.sqrt(np.sum(np.square(trs_opt_corners - curr_corners)) / 4)
        # trs_opt_errors.append(trs_opt_error)

        corner_diff = curr_corners - init_corners
        trans_mean_x = np.mean(corner_diff[0, :])
        trans_mean_y = np.mean(corner_diff[1, :])

        # print 'corner_diff: ', corner_diff
        # print 'trans_mean_x: ', trans_mean_x
        # print 'trans_mean_y: ', trans_mean_y

        trans_mean_mat = getTranslationMatrix(trans_mean_x, trans_mean_y)

        trans_mean_corners_hm = trans_mean_mat * init_corners_hm
        trans_mean_corners = util.dehomogenize(trans_mean_corners_hm)

        # print 'trans_mean_corners: ', trans_mean_corners

        trans_mean_error = math.sqrt(np.sum(np.square(trans_mean_corners - curr_corners)) / 4)
        trans_mean_errors.append(trans_mean_error)

        if (show_img):
            ret, src_img = cap.read()
            if not ret:
                print 'End of sequence reached unexpectedly'
                break
            if show_actual:
                drawRegion(src_img, curr_corners, act_col, 1)
                cv2.putText(src_img, "Actual", (int(curr_corners[0, 0]), int(curr_corners[1, 0])),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, act_col)
            if show_hom:
                drawRegion(src_img, hom_corners, hom_col, 1)
                cv2.putText(src_img, "Homography", (int(hom_corners[0, 0]), int(hom_corners[1, 0])),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, hom_col)
            if show_affine:
                drawRegion(src_img, affine_corners, affine_col, 1)
                cv2.putText(src_img, "Affine", (int(affine_corners[0, 0]), int(affine_corners[1, 0])),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, affine_col)

            if show_sim:
                drawRegion(src_img, sim_corners, sim_col, 1)
                cv2.putText(src_img, "Sim", (int(sim_corners[0, 0]), int(sim_corners[1, 0])),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, sim_col)

            if show_trs:
                drawRegion(src_img, trs_corners, trs_col, 1)
                cv2.putText(src_img, "Trs", (int(trs_corners[0, 0]), int(trs_corners[1, 0])),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, trs_col)

            if show_rt_opt:
                drawRegion(src_img, rt_opt_corners, rt_col, 1)
                cv2.putText(src_img, "RT Opt", (int(rt_opt_corners[0, 0]), int(rt_opt_corners[1, 0])),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, rt_col)

            # if show_trs_opt:
            #     drawRegion(src_img, trs_corners, trs_col, 1)
            #     cv2.putText(src_img, "TrsOpt", (int(trs_opt_corners[0, 0]), int(trs_opt_corners[1, 0])),
            #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, trs_opt_col)

            if show_trans:
                drawRegion(src_img, trans_mean_corners, trs_col, 1)
                cv2.putText(src_img, "Trans", (int(trans_mean_corners[0, 0]), int(trans_mean_corners[1, 0])),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, trs_col)

            cv2.putText(src_img,
                        "8:{:5.2f} 6:{:5.2f} 4:{:5.2f} 3s:{:5.2f} 3so:{:5.2f} 3o:{:5.2f} 2:{:5.2f}".format(
                            hom_error, affine_error, sim_error, trs_error, trs_opt_error, rt_opt_error,
                            trans_mean_error),
                        (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (255, 255, 255))

        if write_corners:
            writeCorners(hom_file, hom_corners, i + 1)
            writeCorners(affine_file, affine_corners, i + 1)
            writeCorners(sim_file, sim_corners, i + 1)
            writeCorners(trs_file, trs_corners, i + 1)
            writeCorners(rt_opt_file, rt_opt_corners, i + 1)
            writeCorners(trans_file, trans_mean_corners, i + 1)

        # if perform_pw_opt:
        #     pw_start_time = time.clock()
        #     # rot_mean_mat = computeRotationLS(trans_mean_corners, curr_corners_norm)
        #     # rot_mean_mat = computeRotationBrute(trans_pts, curr_corners_norm)
        #     # rt_mean_mat = rot_mean_mat * trans_mean_mat
        #     # rt_mean_corners = util.dehomogenize(curr_norm_mat_inv * rt_mean_mat *init_corners_norm_hm)
        #     # rt_mean_error = math.sqrt(np.sum(np.square(rt_mean_corners - curr_corners)) / 4)
        #     # rt_mean_errors.append(rt_mean_error)
        #
        #     # rt_pts = rt_mean_mat * np.mat(util.homogenize(init_corners_norm))
        #     # scale_mean_mat = computeScaleBrute(rt_pts, curr_corners_norm)
        #
        #     # se2_mean_mat = scale_mean_mat * rt_mean_mat
        #     # se2_mean_corners = util.dehomogenize(curr_norm_mat_inv * se2_mean_mat *init_corners_norm_hm)
        #     # se2_mean_error = math.sqrt(np.sum(np.square(se2_mean_corners - curr_corners)) / 4)
        #     # se2_mean_errors.append(se2_mean_error)
        #
        #     rot_opt_mat = computeRotationOpt(trans_mean_corners_hm, curr_corners_norm_hm, opt_method,
        #                                      jacobian_func=rot_jacobian_func)
        #     rt_pwopt_mat = rot_opt_mat * trans_mean_mat
        #
        #     rt_pwopt_corners = util.dehomogenize(curr_norm_mat_inv * rt_pwopt_mat * init_corners_norm_hm)
        #     rt_pwopt_error = math.sqrt(np.sum(np.square(rt_pwopt_corners - curr_corners)) / 4)
        #
        #     rt_pwopt_corners_hm = rt_pwopt_mat * init_corners_norm_hm
        #     scale_opt_mat = computeScaleOpt(rt_pwopt_corners_hm, curr_corners_norm_hm, opt_method,
        #                                     jacobian_func=scale_jacobian_func)
        #
        #     se2_pwopt_mat = scale_opt_mat * rt_pwopt_mat
        #     se2_pwopt_corners = util.dehomogenize(curr_norm_mat_inv * se2_pwopt_mat * init_corners_norm_hm)
        #     se2_pwopt_error = math.sqrt(np.sum(np.square(se2_pwopt_corners - curr_corners)) / 4)
        #
        #     se2_pwopt_corners_hm = se2_pwopt_mat * init_corners_norm_hm
        #
        #     shear_opt_mat = computeSheartOpt(se2_pwopt_corners_hm, curr_corners_norm_hm, opt_method,
        #                                      jacobian_func=shear_jacobian_func)
        #     affine_pwopt_mat = shear_opt_mat * se2_pwopt_mat
        #     affine_pwopt_corners = util.dehomogenize(curr_norm_mat_inv * affine_pwopt_mat * init_corners_norm_hm)
        #     affine_pwopt_error = math.sqrt(np.sum(np.square(affine_pwopt_corners - curr_corners)) / 4)
        #
        #     affine_pwopt_corners_hm = affine_pwopt_mat * init_corners_norm_hm
        #     proj_opt_mat = computeProjOpt(affine_pwopt_corners_hm, curr_corners_norm_hm, opt_method,
        #                                   jacobian_func=proj_jacobian_func)
        #     hom_pwopt_mat = proj_opt_mat * affine_pwopt_mat
        #     hom_pwopt_corners = util.dehomogenize(curr_norm_mat_inv * hom_pwopt_mat * init_corners_norm_hm)
        #     hom_pwopt_error = math.sqrt(np.sum(np.square(hom_pwopt_corners - curr_corners)) / 4)
        #
        #     pw_end_time = time.clock()
        #     pw_curr_fps = 1.0 / (pw_end_time - pw_start_time)
        #
        #     if print_corners:
        #         print 'trans_mean_corners_hm:\n', trans_mean_corners_hm
        #         print 'rt_pwopt_corners_hm:\n', rt_pwopt_corners_hm
        #         print 'se2_pwopt_mat:\n', se2_pwopt_mat
        #         print 'se2_pwopt_corners:\n', se2_pwopt_corners
        #         print 'se2_pwopt_corners_hm:\n', se2_pwopt_corners_hm
        #         print 'curr_corners_norm_hm:\n', curr_corners_norm_hm
        #
        #     if write_corners:
        #         writeCorners(rt_pwopt_file, rt_pwopt_corners)
        #         writeCorners(se2_pwopt_file, se2_pwopt_corners)
        #
        #     if show_trans_mean:
        #         drawRegion(src_img, trans_mean_corners, trans_col, 1)
        #     cv2.putText(src_img, "Trans Mean", (int(trans_mean_corners[0, 0]), int(trans_mean_corners[1, 0])),
        #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, trans_col)
        #     if show_rt_pwopt:
        #         drawRegion(src_img, rt_pwopt_corners, rt_col, 1)
        #     cv2.putText(src_img, "SE2 PW Opt", (int(rt_pwopt_corners[0, 0]), int(rt_pwopt_corners[1, 0])),
        #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, rt_col)
        #     if show_se2_pwopt:
        #         drawRegion(src_img, se2_pwopt_corners, se2_col, 1)
        #     cv2.putText(src_img, "SE2 PW Opt", (int(se2_pwopt_corners[0, 0]), int(se2_pwopt_corners[1, 0])),
        #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, se2_col)

        # rt_pwopt_errors.append(rt_pwopt_error)
        # pw_error_fps.append(pw_curr_fps)
        # se2_pwopt_errors.append(se2_pwopt_error)
        # affine_pwopt_errors.append(affine_pwopt_error)
        # hom_pwopt_errors.append(hom_pwopt_error)

        if perform_sim_opt:

            start_time = time.clock()

            trans_opt_mat = computeTransOpt(init_corners_norm_hm, curr_corners_norm_hm, opt_method,
                                            jacobian_func=trans_jacobian_func)
            trans_opt_corners = util.dehomogenize(curr_norm_mat_inv * trans_opt_mat * init_corners_norm_hm)
            trans_opt_error = math.sqrt(np.sum(np.square(trans_opt_corners - curr_corners)) / 4)

            se2_opt_mat, se2_trans_mat, se2_rot_mat, se2_scale_mat = computeSE2Opt(
                init_corners_norm_hm, curr_corners_norm_hm, opt_method, jacobian_func=se2_jacobian_func)
            se2_opt_corners = util.dehomogenize(curr_norm_mat_inv * se2_opt_mat * init_corners_norm_hm)
            se2_opt_error = math.sqrt(np.sum(np.square(se2_opt_corners - curr_corners)) / 4)

            affine_opt_mat, affine_trans_mat, affine_rot_mat, affine_scale_mat, affine_shear_mat = computeAffineOpt(
                init_corners_norm_hm, curr_corners_norm_hm, opt_method, jacobian_func=affine_jacobian_func)
            affine_opt_corners = util.dehomogenize(curr_norm_mat_inv * affine_opt_mat * init_corners_norm_hm)
            affine_opt_error = math.sqrt(np.sum(np.square(affine_opt_corners - curr_corners)) / 4)

            end_time = time.clock()
            curr_fps = 1.0 / (end_time - start_time)

            # hom_ls_error = math.sqrt(np.sum(np.square(hom_mat_dlt - hom_mat)) / 9)
            if debug_mode:
                print '*' * 100
                print 'frame: ', i
            # affine_mat_inv = np.linalg.inv(affine_mat)
            # proj_fwd_mat = affine_mat_inv * hom_mat
            # proj_inv_mat = hom_mat * affine_mat_inv
            # proj_error = math.sqrt(np.sum(np.square(proj_fwd_mat - proj_inv_mat)) / 9)

            # rt_trans_corners = util.dehomogenize(curr_norm_mat_inv * rt_trans_mat * init_corners_norm_hm)
            # rt_trans_error = math.sqrt(np.sum(np.square(rt_trans_corners - curr_corners)) / 4)
            # se2_trans_corners = util.dehomogenize(curr_norm_mat_inv * se2_trans_mat * init_corners_norm_hm)
            # se2_trans_error = math.sqrt(np.sum(np.square(se2_trans_corners - curr_corners)) / 4)

            if write_corners:
                writeCorners(rt_opt_file, rt_opt_corners)
                writeCorners(se2_opt_file, se2_opt_corners)
            if show_se2_opt:
                drawRegion(src_img, se2_opt_corners, se2_col, 1)
                cv2.putText(src_img, "SE2 Opt", (int(se2_opt_corners[0, 0]), int(se2_opt_corners[1, 0])),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, se2_col)

        se2_opt_errors.append(se2_opt_error)
        affine_opt_errors.append(affine_opt_error)
        trans_opt_errors.append(trans_opt_error)
        rt_trans_errors.append(rt_trans_error)
        se2_trans_errors.append(se2_trans_error)
        error_fps.append(curr_fps)

        if update_init_corners:
            init_corners_norm = np.copy(curr_corners_norm)
            init_corners_norm_hm = np.copy(curr_corners_norm_hm)

        # cv2.putText(src_img,
        #             "hp:{:5.2f} a:{:5.2f} ao:{:5.2f} ap:{:5.2f} se2o:{:5.2f} se2p:{:5.2f} rto:{:5.2f} rtp:{:5.2f}".format(
        #                 hom_pwopt_error,
        #                 affine_error,
        #                 affine_opt_error,
        #                 affine_pwopt_error,
        #                 se2_opt_error,
        #                 se2_pwopt_error),
        #             rt_opt_error,
        #             rt_pwopt_error,
        #             (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (255, 255, 255))
        #
        # cv2.putText(src_img, "pw: {:7.4f} sim: {:7.4f} tm:{:5.2f} to:{:5.2f} tse2:{:5.2f} trt:{:5.2f}".format(
        #     pw_curr_fps, curr_fps,
        #     trans_mean_error,
        #     trans_opt_error,
        #     rt_trans_error,
        #     se2_trans_error),
        #             (5, 40),
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (255, 255, 255))

        if show_img:
            cv2.imshow(window_name, src_img)
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('p'):
                cv2.waitKey(0)
            elif key == ord('8'):
                show_hom = 1 - show_hom
            elif key == ord('6'):
                show_affine = 1 - show_affine
            elif key == ord('4'):
                show_sim = 1 - show_sim
            elif key == ord('3'):
                show_trs = 1 - show_trs
            elif key == ord('2'):
                show_trans = 1 - show_trans
            elif key == ord('m'):
                show_trans_mean = 1 - show_trans_mean
                show_rt_mean = 1 - show_rt_mean

    if write_corners:
        hom_file.close()
        affine_file.close()
        sim_file.close()
        trs_file.close()
        trans_file.close()
        if perform_pw_opt:
            # rt_mean_file.close()
            # se2_mean_file.close()
            rt_opt_file.close()
            rt_pwopt_file.close()
            se2_opt_file.close()
            se2_pwopt_file.close()

    print '\n'
    avg_hom_error = np.mean(hom_errors)
    print 'avg_hom_error: ', avg_hom_error

    avg_affine_error = np.mean(affine_errors)
    print 'avg_affine_error: ', avg_affine_error

    avg_sim_error = np.mean(sim_errors)
    print 'avg_sim_error: ', avg_sim_error

    avg_trs_error = np.mean(trs_errors)
    print 'avg_trs_error: ', avg_trs_error

    avg_trans_error = np.mean(trans_mean_errors)
    print 'avg_trans_error: ', avg_trans_error


    # avg_hom_pwopt_error = np.mean(hom_pwopt_errors)
    # print 'avg_hom_pwopt_error: ', avg_hom_pwopt_error
    #
    # avg_affine_opt_error = np.mean(affine_opt_errors)
    # print 'avg_affine_opt_error: ', avg_affine_opt_error
    #
    # avg_affine_pwopt_error = np.mean(affine_pwopt_errors)
    # print 'avg_affine_pwopt_error: ', avg_affine_pwopt_error
    #
    # avg_trans_mean_error = np.mean(trans_mean_errors)
    # print 'avg_trans_mean_error: ', avg_trans_mean_error
    #
    # avg_trans_opt_error = np.mean(trans_opt_errors)
    # print 'avg_trans_opt_error: ', avg_trans_opt_error
    #
    # avg_rt_trans_error = np.mean(rt_trans_errors)
    # print 'avg_rt_trans_error: ', avg_rt_trans_error
    #
    # avg_se2_trans_error = np.mean(se2_trans_errors)
    # print 'avg_se2_trans_error: ', avg_se2_trans_error
    #
    # # avg_rt_mean_error = np.mean(rt_mean_errors)
    # # print 'avg_rt_mean_error: ', avg_rt_mean_error
    #
    # # avg_se2_mean_error = np.mean(se2_mean_errors)
    # # print 'avg_se2_mean_error: ', avg_se2_mean_error
    #
    # avg_rt_opt_error = np.mean(rt_opt_errors)
    # print 'avg_rt_opt_error: ', avg_rt_opt_error
    #
    # avg_rt_pwopt_error = np.mean(rt_pwopt_errors)
    # print 'avg_rt_pwopt_error: ', avg_rt_pwopt_error
    #
    # avg_se2_opt_error = np.mean(se2_opt_errors)
    # print 'avg_se2_opt_error: ', avg_se2_opt_error
    #
    # avg_se2_pwopt_error = np.mean(se2_pwopt_errors)
    # print 'avg_se2_pwopt_error: ', avg_se2_pwopt_error
    #
    # pw_mean_fps = np.mean(pw_error_fps)
    # print 'pw_mean_fps: ', pw_mean_fps
    #
    # mean_fps = np.mean(error_fps)
    # print 'mean_fps: ', mean_fps
    #
    # errors_combined = np.array([hom_errors, hom_pwopt_errors, affine_opt_errors, affine_pwopt_errors,
    # se2_opt_errors, se2_pwopt_errors, rt_opt_errors, rt_pwopt_errors])
    # np.savetxt('errors_combined.txt', errors_combined.transpose(), fmt='%12.9f', delimiter='\t')
    #
    # res_fname = 'result_gtl.txt'
    # res_file_exists = os.path.isfile(res_fname)
    # result_file = open(res_fname, 'a')
    # if not res_file_exists:
    #     result_file.write(
    #         '{:6s}\t{:20s}\t{:15s}\t{:5s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\n'
    #         .format('seq_id', 'seq_name', 'opt_method', 'prog', 'hom_opt', 'hom_pwopt', 'aff_opt', 'aff_pwopt',
    #                 'se2_opt',
    #                 'se2_pwopt', 'rt_opt', 'rt_pwopt', 'mean_fps', 'pw_mean_fps'))
    #
    #     # result_file.write('seq_id\tseq_name\topt_method\thom\taffine\taffine_opt\taffine_pwopt\t')
    #     # result_file.write('se2_opt\tse2_pwopt\trt_opt\trt_pwopt\t')
    #     # result_file.write('mean_fps\tpw_mean_fps\n')
    # result_file.write(
    #     '{:6d}\t{:20s}\t{:15s}\t{:5d}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{:9.6f}\n'.
    #     format(seq_id, seq_name, opt_method, update_init_corners, avg_hom_error, avg_hom_pwopt_error,
    #            avg_affine_opt_error,
    #            avg_affine_pwopt_error,
    #            avg_se2_opt_error, avg_se2_pwopt_error, avg_rt_opt_error, avg_rt_pwopt_error,
    #            mean_fps, pw_mean_fps))
    # result_file.close()
    #
    # trans_result_file = open('trans_result_gtl.txt', 'a')
    # trans_result_file.write(
    #     '{:02d}\t{:20s}\t{:15s}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{:9.6f}\n'.
    #     format(seq_id, seq_name, opt_method, avg_trans_mean_error, avg_trans_opt_error,
    #            avg_rt_trans_error, avg_se2_trans_error))
    # trans_result_file.close()
    #
    # print '*' * 100












