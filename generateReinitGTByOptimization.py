from optimization import *

if __name__ == '__main__':

    params = {
        'db_root_dir': '../Datasets',
        'gt_root_dir': '../Datasets',
        'img_name_fmt': 'frame%05d.jpg',
        'actor_id': 0,
        'seq_id': 6,
        'opt_id': 2,
        'rearrange_corners': 0,
        'pause_seq': 0,
        'init_frame_id': 0,
        'end_frame_id': 0,
        'show_img': 0,
        'show_actual': 0,
        'show_hom': 0,
        'show_affine': 0,
        'show_sim': 0,
        'show_rt': 0,
        'show_trans': 1,
        'show_trans_rect': 0,
        'generate_actual': 0,
        'generate_hom': 0,
        'generate_affine': 0,
        'generate_sim': 0,
        'generate_rt': 0,
        'generate_trans': 0,
        'generate_trans_rect': 0,
        'font_size': 0.8,
        'update_init_corners': 0,
        'print_corners': 0,
        'write_corners': 1,
        'use_homography_ls': 0,
        'use_affine_ls': 0,
        'use_similarity_ls': 0,
        'use_rt_ls': 1,
        'initialize_with_rect': 0,
        'write_to_bin': 1,
        'gt_col': (0, 255, 0),
        'act_col': (0, 0, 0),
        'hom_col': (0, 0, 255),
        'affine_col': (0, 255, 0),
        'sim_col': (255, 255, 0),
        'rt_col': (255, 0, 0),
        'trans_col': (255, 255, 255),
        'trans_rect_col': (0, 255, 255)
    }
    params = parseArguments(sys.argv, params)

    # settings for synthetic sequences
    syn_ssm = 'c8'
    syn_ssm_sigma_id = 28
    syn_ilm = 'rbf'
    syn_am_sigma_id = 9
    syn_add_noise = 1
    syn_noise_mean = 0
    syn_noise_sigma = 10
    syn_frame_id = 0
    syn_err_thresh = 5.0

    db_root_dir = params['db_root_dir']
    gt_root_dir = params['gt_root_dir']
    img_name_fmt = params['img_name_fmt']
    actor_id = params['actor_id']
    seq_id = params['seq_id']
    opt_id = params['opt_id']
    rearrange_corners = params['rearrange_corners']
    pause_seq = params['pause_seq']
    init_frame_id = params['init_frame_id']
    end_frame_id = params['end_frame_id']
    show_img = params['show_img']
    show_actual = params['show_actual']
    show_hom = params['show_hom']
    show_affine = params['show_affine']
    show_sim = params['show_sim']
    show_rt = params['show_rt']
    show_trans = params['show_trans']
    show_trans_rect = params['show_trans_rect']
    generate_actual = params['generate_actual']
    generate_hom = params['generate_hom']
    generate_affine = params['generate_affine']
    generate_sim = params['generate_sim']
    generate_rt = params['generate_rt']
    generate_trans = params['generate_trans']
    generate_trans_rect = params['generate_trans_rect']
    font_size = params['font_size']
    update_init_corners = params['update_init_corners']
    print_corners = params['print_corners']
    write_corners = params['write_corners']
    use_homography_ls = params['use_homography_ls']
    use_affine_ls = params['use_affine_ls']
    use_similarity_ls = params['use_similarity_ls']
    use_rt_ls = params['use_rt_ls']
    initialize_with_rect = params['initialize_with_rect']
    write_to_bin = params['write_to_bin']
    gt_col = params['gt_col']
    act_col = params['act_col']
    hom_col = params['hom_col']
    affine_col = params['affine_col']
    sim_col = params['sim_col']
    rt_col = params['rt_col']
    trans_col = params['trans_col']
    trans_rect_col = params['trans_rect_col']

    show_actual = show_actual and generate_actual
    show_hom = show_hom and generate_hom
    show_affine = show_affine and generate_affine
    show_sim = show_sim and generate_sim
    show_rt = show_rt and generate_rt
    show_trans = show_trans and generate_trans
    show_trans_rect = show_trans_rect and generate_trans_rect

    if generate_rt and use_rt_ls:
        print 'Using least squares optimization for RT'

    params_dict = getParamDict()
    actors = params_dict['actors']
    sequences = params_dict['sequences']
    opt_methods = params_dict['opt_methods']

    actor = actors[actor_id]
    sequences = sequences[actor]
    seq_name = sequences[seq_id]
    opt_method = opt_methods[opt_id]

    if actor == 'Synthetic':
        seq_name = getSyntheticSeqName(seq_name, syn_ssm, syn_ssm_sigma_id, syn_ilm,
                            syn_am_sigma_id, syn_frame_id, syn_add_noise,
                            syn_noise_mean, syn_noise_sigma)

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

    seq_root_dir = db_root_dir + '/' + actor + '/' + seq_name
    ground_truth_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)
    opt_gt_dir = gt_root_dir + '/' + actor + '/ReinitGT'

    if not write_to_bin:
        opt_gt_dir = opt_gt_dir + '/' + seq_name

    if not os.path.exists(opt_gt_dir):
        os.makedirs(opt_gt_dir)

    print 'actor: ', actor
    print 'seq_id:', seq_id, 'seq_name:', seq_name
    print 'opt_id:', opt_id, 'opt_method:', opt_method
    print 'write_corners:', write_corners
    print 'write_to_bin:', write_to_bin
    print 'gt_root_dir:', gt_root_dir

    no_of_frames = ground_truth.shape[0]
    print 'no_of_frames: ', no_of_frames

    if end_frame_id < init_frame_id:
        end_frame_id = no_of_frames - 1

    out_dir = '{:s}/{:s}/OptGT'.format(gt_root_dir, actor)
    if init_frame_id == 0 and init_frame_id == end_frame_id:
        write_to_bin = 0
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    id_permutations = [np.array([0, 1, 2, 3], dtype=np.uint32),
                       np.array([3, 0, 1, 2], dtype=np.uint32),
                       np.array([2, 3, 0, 1], dtype=np.uint32),
                       np.array([1, 2, 3, 0], dtype=np.uint32)]

    if write_corners and write_to_bin:
        if generate_actual:
            act_fname = '{:s}/{:s}.bin'.format(opt_gt_dir, seq_name)
            act_file = open(act_fname, 'wb')
            np.array([end_frame_id + 1], dtype=np.uint32).tofile(act_file)

        if generate_hom:
            hom_fname = '{:s}/{:s}_8.bin'.format(opt_gt_dir, seq_name)
            hom_file = open(hom_fname, 'wb')
            np.array([end_frame_id + 1], dtype=np.uint32).tofile(hom_file)

        if generate_affine:
            affine_fname = '{:s}/{:s}_6.bin'.format(opt_gt_dir, seq_name)
            affine_file = open(affine_fname, 'wb')
            np.array([end_frame_id + 1], dtype=np.uint32).tofile(affine_file)

        if generate_sim:
            sim_fname = '{:s}/{:s}_4.bin'.format(opt_gt_dir, seq_name)
            sim_file = open(sim_fname, 'wb')
            np.array([end_frame_id + 1], dtype=np.uint32).tofile(sim_file)

        if generate_rt:
            rt_fname = '{:s}/{:s}_3.bin'.format(opt_gt_dir, seq_name)
            rt_file = open(rt_fname, 'wb')
            np.array([end_frame_id + 1], dtype=np.uint32).tofile(rt_file)

        if generate_trans:
            trans_fname = '{:s}/{:s}_2.bin'.format(opt_gt_dir, seq_name)
            trans_file = open(trans_fname, 'wb')
            np.array([end_frame_id + 1], dtype=np.uint32).tofile(trans_file)

        if generate_trans_rect:
            trans_rect_fname = '{:s}/{:s}_2r.bin'.format(opt_gt_dir, seq_name)
            trans_rect_file = open(trans_rect_fname, 'wb')
            np.array([end_frame_id + 1], dtype=np.uint32).tofile(trans_rect_file)

    overall_start_time = time.clock()
    for start_id in xrange(init_frame_id, end_frame_id + 1):
        print 'seq {:d}: {:s} frame: {:d} of {:d}'.format(
            seq_id, seq_name, start_id + 1, end_frame_id + 1)
        init_corners = np.asarray([ground_truth[start_id, 0:2].tolist(),
                                   ground_truth[start_id, 2:4].tolist(),
                                   ground_truth[start_id, 4:6].tolist(),
                                   ground_truth[start_id, 6:8].tolist()]).T
        if rearrange_corners:
            print 'init_corners:\n', init_corners
            rearranged_corners, rearrangement_ids = arrangeCornersWithIDs(init_corners)
            print 'rearranged_corners:\n', rearranged_corners
            print 'rearrangement_ids:\n', rearrangement_ids
            rearranged_corners2 = init_corners[:, rearrangement_ids]
            print 'rearranged_corners2:\n', rearranged_corners2
            init_corners = rearranged_corners.copy()

            if np.array_equal(rearrangement_ids, id_permutations[0]) \
                    or np.array_equal(rearrangement_ids, id_permutations[1]) \
                    or np.array_equal(rearrangement_ids, id_permutations[2]) \
                    or np.array_equal(rearrangement_ids, id_permutations[3]):
                pass
            else:
                print 'rearrangement_ids: ', rearrangement_ids
                raise StandardError('Invalid permutation found')

        init_img = cv2.imread('{:s}/frame{:05d}.jpg'.format(seq_root_dir, start_id + 1))
        if initialize_with_rect:
            init_corners = getRectangularApproximation(init_corners, init_img.shape)

        # (init_corners_norm, base_norm_mat_inv) = shiftGeometricCenterToOrigin(init_corners)
        (init_corners_norm, base_norm_mat_inv) = conditioning_func(init_corners)
        init_corners_hm = np.mat(util.homogenize(init_corners))
        init_corners_norm_hm = np.mat(util.homogenize(init_corners_norm))
        # print 'init_corners:\n', init_corners
        # print 'init_corners2:\n', init_corners2
        init_corners_rect = getRectangularApproximation(init_corners, init_img.shape)
        init_corners_rect_hm = np.mat(util.homogenize(init_corners_rect))
        # print 'init_corners_rect:\n', init_corners_rect
        # print 'init_corners_rect_hm:\n', init_corners_rect_hm
        if show_img:
            window_name = 'seq {:d}: {:s} frame{:05d}'.format(seq_id, seq_name, start_id + 1)
            cv2.namedWindow(window_name)


        # drawRegion(init_img, init_corners_rect, (0, 255, 0), 1)
        # cv2.putText(init_img, "Corners", (int(init_corners_rect[0, 0]), int(init_corners_rect[1, 0])),
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (0, 255, 0))
        # drawRegion(init_img, init_corners_rect2, (255, 0, 0), 1)
        # cv2.putText(init_img, "Centroid", (int(init_corners_rect2[0, 0]), int(init_corners_rect2[1, 0])),
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (255, 0, 0))
        # cv2.imshow(window_name, init_img)
        # cv2.waitKey(0)

        hom_errors = []
        affine_errors = []
        sim_errors = []
        rt_errors = []
        # trs_errors = []
        trans_mean_errors = []
        trans_rect_mean_errors = []
        error_fps = []

        hom_error = 0
        affine_error = 0
        trans_mean_error = 0
        rt_mean_error = 0
        rt_trans_error = 0
        se2_mean_error = 0
        trans_opt_error = 0
        affine_opt_error = 0
        se2_opt_error = 0
        se2_trans_error = 0
        rt_error = 0
        sim_error = 0

        curr_fps = 0
        pw_curr_fps = 0

        if write_corners:
            if write_to_bin:
                if generate_actual:
                    init_corners.astype(np.float64).tofile(act_file)
                if generate_hom:
                    init_corners.astype(np.float64).tofile(hom_file)
                if generate_affine:
                    init_corners.astype(np.float64).tofile(affine_file)
                if generate_sim:
                    init_corners.astype(np.float64).tofile(sim_file)
                if generate_rt:
                    init_corners.astype(np.float64).tofile(rt_file)
                if generate_trans:
                    init_corners.astype(np.float64).tofile(trans_file)
                if generate_trans_rect:
                    init_corners_rect.astype(np.float64).tofile(trans_rect_file)
            else:
                if generate_actual:
                    if init_frame_id == 0 and init_frame_id == end_frame_id:
                        act_fname = '{:s}/{:s}.txt'.format(out_dir, seq_name)
                    else:
                        act_fname = '{:s}/frame{:05d}.txt'.format(opt_gt_dir, start_id + 1)
                    act_file = open(act_fname, 'w')
                    writeCorners(act_file, init_corners, start_id + 1, 1)

                if generate_hom:
                    if init_frame_id == 0 and init_frame_id == end_frame_id:
                        hom_fname = '{:s}/{:s}_8.txt'.format(out_dir, seq_name)
                    else:
                        hom_fname = '{:s}/frame{:05d}_8.txt'.format(opt_gt_dir, start_id + 1)
                    hom_file = open(hom_fname, 'w')
                    writeCorners(hom_file, init_corners, start_id + 1, 1)

                if generate_affine:
                    if init_frame_id == 0 and init_frame_id == end_frame_id:
                        affine_fname = '{:s}/{:s}_6.txt'.format(out_dir, seq_name)
                    else:
                        affine_fname = '{:s}/frame{:05d}_6.txt'.format(opt_gt_dir, start_id + 1)
                    affine_file = open(affine_fname, 'w')
                    writeCorners(affine_file, init_corners, start_id + 1, 1)

                if generate_sim:
                    if init_frame_id == 0 and init_frame_id == end_frame_id:
                        sim_fname = '{:s}/{:s}_4.txt'.format(out_dir, seq_name)
                    else:
                        sim_fname = '{:s}/frame{:05d}_4.txt'.format(opt_gt_dir, start_id + 1)
                    sim_file = open(sim_fname, 'w')
                    writeCorners(sim_file, init_corners, start_id + 1, 1)

                # trs_fname = '{:s}/frame{:05d}_3s.txt'.format(opt_gt_dir, start_id + 1)
                # trs_file = open(trs_fname, 'w')
                # writeCorners(trs_file, init_corners, start_id + 1, 1)

                if generate_rt:
                    if init_frame_id == 0 and init_frame_id == end_frame_id:
                        rt_fname = '{:s}/{:s}_3.txt'.format(out_dir, seq_name)
                    else:
                        rt_fname = '{:s}/frame{:05d}_3.txt'.format(opt_gt_dir, start_id + 1)
                    rt_file = open(rt_fname, 'w')
                    writeCorners(rt_file, init_corners, start_id + 1, 1)

                if generate_trans:
                    if init_frame_id == 0 and init_frame_id == end_frame_id:
                        trans_fname = '{:s}/{:s}_2.txt'.format(out_dir, seq_name)
                    else:
                        trans_fname = '{:s}/frame{:05d}_2.txt'.format(opt_gt_dir, start_id + 1)
                    trans_file = open(trans_fname, 'w')
                    writeCorners(trans_file, init_corners, start_id + 1, 1)

                if generate_trans_rect:
                    if init_frame_id == 0 and init_frame_id == end_frame_id:
                        trans_rect_fname = '{:s}/{:s}_2r.txt'.format(out_dir, seq_name)
                    else:
                        trans_rect_fname = '{:s}/frame{:05d}_2r.txt'.format(opt_gt_dir, start_id + 1)
                    trans_rect_file = open(trans_rect_fname, 'w')
                    writeCorners(trans_rect_file, init_corners_rect, start_id + 1, 1)

        states = ['off', 'on']
        start_time = time.clock()
        for i in xrange(start_id + 1, no_of_frames):

            curr_corners = np.asarray([ground_truth[i, 0:2].tolist(),
                                       ground_truth[i, 2:4].tolist(),
                                       ground_truth[i, 4:6].tolist(),
                                       ground_truth[i, 6:8].tolist()]).T
            if rearrange_corners:
                curr_corners = curr_corners[:, rearrangement_ids]

            (curr_corners_norm, curr_norm_mat_inv) = conditioning_func(curr_corners)
            curr_corners_hm = util.homogenize(curr_corners)
            curr_corners_norm_hm = util.homogenize(curr_corners_norm)
            # print 'curr_corners:\n', curr_corners
            # print 'curr_corners_norm:\n', curr_corners_norm

            if use_homography_ls:
                hom_mat = computeHomographyLS(init_corners_norm, curr_corners_norm)
            else:
                hom_mat = np.mat(util.compute_homography(init_corners_norm, curr_corners_norm))

            if generate_hom:
                hom_corners = util.dehomogenize(curr_norm_mat_inv * hom_mat * init_corners_norm_hm)
                hom_error = math.sqrt(np.sum(np.square(hom_corners - curr_corners)) / 4)
                hom_errors.append(hom_error)

            if generate_affine:
                affine_mat = computeAffineLS(init_corners, curr_corners)
                affine_corners = util.dehomogenize(affine_mat * init_corners_hm)
                affine_error = math.sqrt(np.sum(np.square(affine_corners - curr_corners)) / 4)
                affine_errors.append(affine_error)

            if generate_sim:
                sim_mat = computeSimilarityLS(init_corners, curr_corners)
                sim_corners = util.dehomogenize(sim_mat * init_corners_hm)
                sim_error = math.sqrt(np.sum(np.square(sim_corners - curr_corners)) / 4)
                sim_errors.append(sim_error)

            # trs_mat = computeTranscalingyLS(init_corners, curr_corners)
            # trs_corners = util.dehomogenize(trs_mat * init_corners_hm)
            # trs_error = math.sqrt(np.sum(np.square(trs_corners - curr_corners)) / 4)
            # trs_errors.append(trs_error)

            if generate_rt:
                if use_rt_ls:
                    sim_mat = computeSimilarityLS(init_corners, curr_corners)
                    a = sim_mat[0, 0] - 1
                    b = sim_mat[1, 0]
                    theta = np.arctan2 (b, a + 1)
                    cos_theta = math.cos(theta)
                    sin_theta = math.sin(theta)
                    rt_mat = np.mat(
                        [[cos_theta, - sin_theta, sim_mat[0, 2]],
                         [sin_theta, cos_theta, sim_mat[1, 2]],
                         [0, 0, 1]]
                    )
                else:
                    rt_mat, rt_trans_mat, rt_rot_mat = computeRTOpt(init_corners_hm, curr_corners_hm, opt_method,
                                                                    jacobian_func=rt_jacobian_func)
                rt_corners = util.dehomogenize(rt_mat * init_corners_hm)
                rt_error = math.sqrt(np.sum(np.square(rt_corners - curr_corners)) / 4)
                rt_errors.append(rt_error)

            if generate_trans:
                corner_diff = curr_corners - init_corners
                trans_mean_x = np.mean(corner_diff[0, :])
                trans_mean_y = np.mean(corner_diff[1, :])
                trans_mean_mat = getTranslationMatrix(trans_mean_x, trans_mean_y)
                trans_mean_corners_hm = trans_mean_mat * init_corners_hm
                trans_mean_corners = util.dehomogenize(trans_mean_corners_hm)
                trans_mean_error = math.sqrt(np.sum(np.square(trans_mean_corners - curr_corners)) / 4)
                trans_mean_errors.append(trans_mean_error)

            if generate_trans_rect:
                corner_diff = curr_corners - init_corners_rect
                trans_mean_x = np.mean(corner_diff[0, :])
                trans_mean_y = np.mean(corner_diff[1, :])
                trans_mean_mat = getTranslationMatrix(trans_mean_x, trans_mean_y)
                trans_rect_mean_corners_hm = trans_mean_mat * init_corners_rect_hm
                trans_rect_mean_corners = util.dehomogenize(trans_rect_mean_corners_hm)
                trans_rect_mean_error = math.sqrt(np.sum(np.square(trans_rect_mean_corners - curr_corners)) / 4)
                trans_rect_mean_errors.append(trans_rect_mean_error)
                # print 'trans_mean_corners: ', trans_mean_corners

            if show_img:
                src_img = cv2.imread('{:s}/frame{:05d}.jpg'.format(seq_root_dir, i + 1))
                if src_img is None:
                    raise StandardError('End of sequence reached unexpectedly')

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

                # if show_trs:
                # drawRegion(src_img, trs_corners, trs_col, 1)
                # cv2.putText(src_img, "Trs", (int(trs_corners[0, 0]), int(trs_corners[1, 0])),
                # cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, trs_col)

                if show_rt:
                    drawRegion(src_img, rt_corners, rt_col, 1)
                    cv2.putText(src_img, "RT Opt", (int(rt_corners[0, 0]), int(rt_corners[1, 0])),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, rt_col)
                if show_trans:
                    drawRegion(src_img, trans_mean_corners, trans_col, 1)
                    cv2.putText(src_img, "Trans", (int(trans_mean_corners[0, 0]), int(trans_mean_corners[1, 0])),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, trans_col)
                if show_trans_rect:
                    drawRegion(src_img, trans_rect_mean_corners, trans_col, 1)
                    cv2.putText(src_img, "TransR",
                                (int(trans_rect_mean_corners[0, 0]), int(trans_rect_mean_corners[1, 0])),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, trans_rect_col)
                cv2.putText(src_img,
                            "8:{:5.2f} 6:{:5.2f} 4:{:5.2f} 3:{:5.2f} 2:{:5.2f}".format(
                                hom_error, affine_error, sim_error, rt_error, trans_mean_error),
                            (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (255, 255, 255))

            if write_corners:
                if write_to_bin:
                    if generate_actual:
                        curr_corners.astype(np.float64).tofile(act_file)
                    if generate_hom:
                        hom_corners.astype(np.float64).tofile(hom_file)
                    if generate_affine:
                        affine_corners.astype(np.float64).tofile(affine_file)
                    if generate_sim:
                        sim_corners.astype(np.float64).tofile(sim_file)
                    if generate_rt:
                        rt_corners.astype(np.float64).tofile(rt_file)
                    if generate_trans:
                        trans_mean_corners.astype(np.float64).tofile(trans_file)
                    if generate_trans_rect:
                        trans_rect_mean_corners.astype(np.float64).tofile(trans_rect_file)
                else:
                    if generate_actual:
                        writeCorners(act_file, curr_corners, i + 1)
                    if generate_hom:
                        writeCorners(hom_file, hom_corners, i + 1)
                    if generate_affine:
                        writeCorners(affine_file, affine_corners, i + 1)
                    if generate_sim:
                        writeCorners(sim_file, sim_corners, i + 1)
                    # writeCorners(trs_file, trs_corners, i + 1)
                    if generate_rt:
                        writeCorners(rt_file, rt_corners, i + 1)
                    if generate_trans:
                        writeCorners(trans_file, trans_mean_corners, i + 1)
                    if generate_trans_rect:
                        writeCorners(trans_rect_file, trans_rect_mean_corners, i + 1)

            error_fps.append(curr_fps)

            if update_init_corners:
                init_corners_norm = np.copy(curr_corners_norm)
                init_corners_norm_hm = np.copy(curr_corners_norm_hm)

            if show_img:
                cv2.imshow(window_name, src_img)
                key = cv2.waitKey(1 - pause_seq)
                if key == 27:
                    break
                elif key == ord('p') or key == 32:
                    pause_seq = 1 - pause_seq
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

        if show_img:
            cv2.destroyWindow(window_name)

        if write_corners and not write_to_bin:
            if generate_actual:
                act_file.close()
            if generate_hom:
                hom_file.close()
            if generate_affine:
                affine_file.close()
            if generate_sim:
                sim_file.close()
            # trs_file.close()
            if generate_rt:
                rt_file.close()
            if generate_trans:
                trans_file.close()
            if generate_trans_rect:
                trans_rect_file.close()

        end_time = time.clock()
        curr_time = end_time - start_time
        print 'Time taken: {:f} secs'.format(curr_time)

        if generate_hom:
            avg_hom_error = np.mean(hom_errors)
            print 'avg_hom_error: ', avg_hom_error
        if generate_affine:
            avg_affine_error = np.mean(affine_errors)
            print 'avg_affine_error: ', avg_affine_error
        if generate_sim:
            avg_sim_error = np.mean(sim_errors)
            print 'avg_sim_error: ', avg_sim_error
        # avg_trs_error = np.mean(trs_errors)
        # print 'avg_trs_error: ', avg_trs_error
        if generate_rt:
            avg_rt_error = np.mean(rt_errors)
            print 'avg_rt_error: ', avg_rt_error
        if generate_trans:
            avg_trans_error = np.mean(trans_mean_errors)
            print 'avg_trans_error: ', avg_trans_error
        if generate_trans_rect:
            avg_trans_rect_error = np.mean(trans_rect_mean_errors)
            print 'avg_trans_rect_error: ', avg_trans_rect_error
        print '\n'

    overall_end_time = time.clock()
    overall_curr_time = overall_end_time - overall_start_time
    print 'Total time taken: {:f} secs'.format(overall_curr_time)
    if write_corners and write_to_bin:
        if generate_actual:
            act_file.close()
        if generate_hom:
            hom_file.close()
        if generate_affine:
            affine_file.close()
        if generate_sim:
            sim_file.close()
        if generate_rt:
            rt_file.close()
        if generate_trans:
            trans_file.close()
        if generate_trans_rect:
            trans_rect_file.close()












