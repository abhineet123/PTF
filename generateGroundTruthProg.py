from decomposition import *
from Misc import getBinaryPtsImage2

import time

if __name__ == '__main__':

    sequences = {0: 'nl_bookI_s3',
                 1: 'nl_bookII_s3',
                 2: 'nl_bookIII_s3',
                 3: 'nl_bus',
                 4: 'nl_cereal_s3',
                 5: 'nl_highlighting',
                 6: 'nl_juice_s3',
                 7: 'nl_letter',
                 8: 'nl_mugI_s3',
                 9: 'nl_mugII_s3',
                 10: 'nl_mugIII_s3',
                 11: 'nl_newspaper',
                 12: 'nl_mugII_s1',
                 13: 'nl_mugII_s2',
                 14: 'nl_mugII_s4',
                 15: 'nl_mugII_s5',
                 16: 'nl_mugII_si',
                 17: 'dl_mugII_s1',
                 18: 'dl_mugII_s2',
                 19: 'dl_mugII_s3',
                 20: 'dl_mugII_s4',
                 21: 'dl_mugII_s5',
                 22: 'dl_mugII_si'
    }
    db_root_path = 'E:/UofA/Thesis/#Code/Datasets'
    actor = 'Human'
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d.jpg'
    seq_id = 11
    # flags
    update_base_corners = 0
    use_homography_ls = 1
    use_affine_ls = 0
    use_inverse_decomposition = 0
    initialize_with_rect = 1
    use_jaccard_error = 0

    std_resx = 10
    std_resy = 10

    arg_id = 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        use_affine_ls = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        use_inverse_decomposition = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        use_homography_ls = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        initialize_with_rect = int(sys.argv[arg_id])
        arg_id += 1

    if seq_id >= len(sequences):
        print 'Invalid dataset_id: ', seq_id
        sys.exit()

    show_actual = 0
    show_hom = 1
    show_affine = 1
    show_se2 = 1
    show_rt = 1
    show_trans = 1
    show_trans_mean = 0

    seq_name = sequences[seq_id]

    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    print 'use_affine_ls: ', use_affine_ls
    print 'use_inverse_decomposition: ', use_inverse_decomposition
    print 'use_homography_ls: ', use_homography_ls
    print 'update_base_corners: ', update_base_corners

    src_fname = db_root_path + '/' + actor + '/' + seq_name + '/' + img_name_fmt
    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        print 'The video file ', src_fname, ' could not be opened'
        sys.exit()

    ground_truth_fname = db_root_path + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)
    no_of_frames = ground_truth.shape[0]
    print 'no_of_frames: ', no_of_frames

    base_corners = np.asarray([ground_truth[0, 0:2].tolist(),
                               ground_truth[0, 2:4].tolist(),
                               ground_truth[0, 4:6].tolist(),
                               ground_truth[0, 6:8].tolist()]).T
    if initialize_with_rect:
        base_corners = getRectangularApproximation(base_corners)

    # (base_corners_norm, base_norm_mat_inv) = shiftGeometricCenterToOrigin(base_corners)
    (base_corners_norm, base_norm_mat_inv) = getNormalizedPoints(base_corners)
    base_corners_hm = np.mat(util.homogenize(base_corners_norm))
    base_mean_dist = np.mean(np.sqrt(np.sum(np.square(base_corners_norm), axis=0)))
    # print 'base_corners:\n', base_corners
    # print 'base_corners_norm:\n', base_corners_norm
    # print 'base_norm_mat_inv:\n', base_norm_mat_inv
    # print 'base_mean_dist: ', base_mean_dist

    hom_fname = db_root_path + '/' + actor + '/' + seq_name + '_hom.txt'
    affine_fname = db_root_path + '/' + actor + '/' + seq_name + '_affine.txt'
    se2_fname = db_root_path + '/' + actor + '/' + seq_name + '_se2.txt'
    rt_fname = db_root_path + '/' + actor + '/' + seq_name + '_rt.txt'
    trans_fname = db_root_path + '/' + actor + '/' + seq_name + '_trans.txt'

    hom_file = open(hom_fname, 'w')
    affine_file = open(affine_fname, 'w')
    se2_file = open(se2_fname, 'w')
    rt_file = open(rt_fname, 'w')
    trans_file = open(trans_fname, 'w')

    writeCorners(hom_file, base_corners)
    writeCorners(affine_file, base_corners)
    writeCorners(se2_file, base_corners)
    writeCorners(rt_file, base_corners)
    writeCorners(trans_file, base_corners)

    curr_bin_window_name = 'Current Position'
    affine_bin_window_name = 'Affine Position'
    se2_bin_window_name = 'SE2 Position'
    rt_bin_window_name = 'RT Position'

    ret, init_img = cap.read()
    window_name = 'Homography Decomposition'
    cv2.namedWindow(window_name)
    img_shape = init_img.shape

    # std_pts, std_corners = getNormalizedUnitSquarePts(std_resx, std_resy)
    # std_corners_hm = util.homogenize(std_corners)
    # print 'std_corners:\n', std_corners
    # print 'std_corners_hm:\n', std_corners_hm

    # base_hom_mat = np.mat(util.compute_homography(std_corners, base_corners_norm))
    # # base_pts_norm = util.dehomogenize(base_hom_mat * util.homogenize(std_pts))
    # # base_pts = util.dehomogenize(base_norm_mat_inv * util.homogenize(base_pts_norm))
    #
    #
    # if use_affine_ls:
    #     base_affine_mat = computeAffineLS(std_corners, base_corners_norm)
    # else:
    #     if use_inverse_decomposition:
    #         (base_affine_mat, base_proj_mat, base_hom_rec_mat) = decomposeHomographyInverse(base_hom_mat)
    #     else:
    #         (base_affine_mat, base_proj_mat, base_hom_rec_mat) = decomposeHomographyForward(base_hom_mat)
    # print 'base_affine_mat:\n', base_affine_mat
    # if use_inverse_decomposition:
    #     (base_trans_mat, base_rt_mat, base_se2_mat, base_affine_rec_mat, base_affine_params) = decomposeAffineInverse(
    #         base_affine_mat)
    # else:
    #     (base_trans_mat, base_rt_mat, base_se2_mat, base_affine_rec_mat, base_affine_params) = decomposeAffineForward(
    #         base_affine_mat)

    if use_jaccard_error:
        # base_img = getBinaryPtsImage(img_shape, base_pts)
        base_img = getBinaryPtsImage2(img_shape, base_corners)
        cv2.imshow(curr_bin_window_name, base_img)
    # np.savetxt('std_pts.txt', std_pts.transpose(), fmt='%10.6f', delimiter='\t')
    # np.savetxt('base_pts.txt', base_pts.transpose(), fmt='%10.6f', delimiter='\t')

    act_col = (0, 0, 0)
    hom_col = (0, 0, 255)
    affine_col = (0, 255, 0)
    se2_col = (255, 255, 0)
    rt_col = (255, 0, 0)
    trans_col = (255, 255, 255)

    hom_errors = []
    affine_errors = []
    affine_rec_errors = []
    se2_errors = []
    rt_errors = []
    trans_errors = []
    trans_mean_errors = []

    error_fps = []

    # intras_affine_errors = []
    # affine_inv_errors = []
    # affine_fwd_errors = []
    # affine_mean_errors = []

    prev_corners_norm=base_corners_norm.copy()

    states = ['off', 'on']

    for i in xrange(1, no_of_frames):


        curr_corners = np.asarray([ground_truth[i, 0:2].tolist(),
                                   ground_truth[i, 2:4].tolist(),
                                   ground_truth[i, 4:6].tolist(),
                                   ground_truth[i, 6:8].tolist()]).T

        # curr_corners_norm, curr_norm_mat_inv = getNormalizedPoints(curr_corners)
        # (curr_corners_norm, curr_norm_mat_inv) = shiftProjectiveCenterToOrigin(curr_corners)
        curr_mean_dist = np.mean(np.sqrt(np.sum(np.square(curr_corners_norm), axis=0)))
        curr_centroid = np.mean(curr_corners_norm, axis=1)
        # print 'curr_corners:\n', curr_corners
        # print 'curr_corners_norm:\n', curr_corners_norm
        # print 'curr_norm_mat_inv:\n', curr_norm_mat_inv
        # print 'curr_centroid: ', curr_centroid
        # print 'curr_mean_dist: ', curr_mean_dist
        if use_homography_ls:
            hom_mat = computeHomographyLS(base_corners_norm, curr_corners_norm)
        else:
            hom_mat = np.mat(util.compute_homography(base_corners_norm, curr_corners_norm))


        # np.savetxt('curr_pts.txt', curr_pts.transpose(), fmt='%10.6f', delimiter='\t')
        if use_affine_ls:
            affine_mat = computeAffineLS(base_corners_norm, curr_corners_norm)
            # affine_mat_inv = np.linalg.inv(affine_mat)
            # proj_fwd_mat = affine_mat_inv * hom_mat
            # proj_inv_mat = hom_mat * affine_mat_inv
            # proj_error = math.sqrt(np.sum(np.square(proj_fwd_mat - proj_inv_mat)) / 9)
            # print 'proj_fwd_mat:\n', proj_fwd_mat
            # print 'proj_inv_mat:\n', proj_inv_mat
            # print 'proj_error:', proj_error
        else:
            if use_inverse_decomposition:
                # (affine_mat, proj_mat, hom_rec_mat) = decomposeHomographyMean(hom_mat)
                (affine_mat, proj_mat, hom_rec_mat) = decomposeHomographyInverse(hom_mat)
            else:
                (affine_mat, proj_mat, hom_rec_mat) = decomposeHomographyForward(hom_mat)
        if use_inverse_decomposition:
            (trans_mat, rt_mat, se2_mat, affine_rec_mat, affine_params) = decomposeAffineInverse(affine_mat)
        else:
            (trans_mat, rt_mat, se2_mat, affine_rec_mat, affine_params) = decomposeAffineForward(affine_mat)

        # se2_params = getAugmentingSE2Params(base_affine_params[0:4], affine_params[0:4])
        # (trans_mat, rot_mat, scale_mat) = getDecomposedSE2Matrices(se2_params)
        # if use_inverse_decomposition:
        #     rt_mat = rot_mat * trans_mat
        #     se2_mat = scale_mat * rt_mat
        # else:
        #     rt_mat = trans_mat * rot_mat
        #     se2_mat = trans_mat * rot_mat * scale_mat

        corner_diff = curr_corners_norm - base_corners_norm
        tx = np.mean(corner_diff[0, :])
        ty = np.mean(corner_diff[1, :])
        trans_mean_mat = getTranslationMatrix(tx, ty)

        hom_corners = util.dehomogenize(curr_norm_mat_inv * hom_mat * base_corners_hm)
        affine_corners = util.dehomogenize(curr_norm_mat_inv * affine_mat * base_corners_hm)
        se2_corners = util.dehomogenize(curr_norm_mat_inv * se2_mat * base_corners_hm)
        rt_corners = util.dehomogenize(curr_norm_mat_inv * rt_mat * base_corners_hm)
        trans_corners = util.dehomogenize(curr_norm_mat_inv * trans_mat * base_corners_hm)
        trans_mean_corners = util.dehomogenize(curr_norm_mat_inv * trans_mean_mat * base_corners_hm)

        start_time = time.clock()
        if use_jaccard_error:
            # curr_pts = util.dehomogenize(curr_norm_mat_inv * hom_mat * util.homogenize(std_pts))
            # curr_img = getBinaryPtsImage(init_img.shape, curr_pts)
            curr_img = getBinaryPtsImage2(init_img.shape, curr_corners)
            cv2.imshow(curr_bin_window_name, curr_img)

            # affine_pts = util.dehomogenize(curr_norm_mat_inv * affine_mat * util.homogenize(std_pts))
            # affine_img = getBinaryPtsImage(init_img.shape, affine_pts)
            affine_img = getBinaryPtsImage2(init_img.shape, affine_corners)
            affine_error = getJaccardError(affine_img, curr_img)
            cv2.imshow(affine_bin_window_name, affine_img)

            # se2_pts = util.dehomogenize(curr_norm_mat_inv * se2_mat * util.homogenize(base_pts_norm))
            # se2_img = getBinaryPtsImage(img_shape, se2_pts)
            se2_img = getBinaryPtsImage2(img_shape, se2_corners)
            se2_error = getJaccardError(se2_img, curr_img)

            # rt_pts = util.dehomogenize(curr_norm_mat_inv * rt_mat * util.homogenize(base_pts_norm))
            # rt_img = getBinaryPtsImage(img_shape, rt_pts)
            rt_img = getBinaryPtsImage2(img_shape, rt_corners)
            rt_error = getJaccardError(rt_img, curr_img)
            cv2.imshow(rt_bin_window_name, rt_img)

            # trans_pts = util.dehomogenize(curr_norm_mat_inv * trans_mat * util.homogenize(base_pts_norm))
            # trans_img = getBinaryPtsImage(img_shape, trans_pts)
            trans_img = getBinaryPtsImage2(img_shape, trans_corners)
            trans_error = getJaccardError(trans_img, curr_img)

            # trans_mean_pts = util.dehomogenize(curr_norm_mat_inv * trans_mean_mat * util.homogenize(base_pts_norm))
            # trans_mean_img = getBinaryPtsImage(init_img.shape, trans_mean_pts)
            trans_mean_img = getBinaryPtsImage2(init_img.shape, trans_mean_corners)
            trans_mean_error = getJaccardError(trans_mean_img, curr_img)

            cv2.imshow(se2_bin_window_name, se2_img)
        else:
            affine_error = math.sqrt(np.sum(np.square(affine_corners - curr_corners)) / 4)
            se2_error = math.sqrt(np.sum(np.square(se2_corners - curr_corners)) / 4)
            rt_error = math.sqrt(np.sum(np.square(rt_corners - curr_corners)) / 4)
            trans_error = math.sqrt(np.sum(np.square(trans_corners - curr_corners)) / 4)
            trans_mean_error = math.sqrt(np.sum(np.square(trans_mean_corners - curr_corners)) / 4)
        end_time = time.clock()

        hom_error = math.sqrt(np.sum(np.square(hom_corners - curr_corners)) / 4)

        hom_errors.append(hom_error)
        affine_errors.append(affine_error)
        se2_errors.append(se2_error)
        rt_errors.append(rt_error)
        trans_errors.append(trans_error)
        trans_mean_errors.append(trans_mean_error)

        curr_fps = 1.0 / (end_time - start_time)
        error_fps.append(curr_fps)
        # print 'affine_corners_hm: ', affine_corners_hm
        # print 'base_corners:\n', base_corners
        # print 'curr_corners: ', curr_corners
        # print 'hom_corners_hm:\n', hom_corners_hm
        # print 'hom_corners:\n', hom_corners
        # print 'affine_corners:\n', affine_corners
        # print 'rt_corners:\n', rt_corners
        # print 'trans_corners:\n', trans_corners

        ret, src_img = cap.read()
        if not ret:
            print 'End of sequence reached unexpectedly'
            break

        writeCorners(hom_file, hom_corners)
        writeCorners(affine_file, affine_corners)
        writeCorners(se2_file, se2_corners)
        writeCorners(rt_file, rt_corners)
        writeCorners(trans_file, trans_mean_corners)

        if show_actual:
            drawRegion(src_img, curr_corners, act_col, 1)
            cv2.putText(src_img, "Actual", (int(curr_corners[0, 0]), int(curr_corners[1, 0])),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, act_col)
        if show_hom:
            drawRegion(src_img, hom_corners, hom_col, 1)
            cv2.putText(src_img, "Homography", (int(hom_corners[0, 0]), int(hom_corners[1, 0])),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, hom_col)
        if show_affine:
            drawRegion(src_img, affine_corners, affine_col, 1)
            cv2.putText(src_img, "Affine", (int(affine_corners[0, 0]), int(affine_corners[1, 0])),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, affine_col)
        if show_se2:
            drawRegion(src_img, se2_corners, se2_col, 1)
            cv2.putText(src_img, "SE2", (int(se2_corners[0, 0]), int(se2_corners[1, 0])),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, se2_col)
        if show_rt:
            drawRegion(src_img, rt_corners, rt_col, 1)
            cv2.putText(src_img, "RT", (int(rt_corners[0, 0]), int(rt_corners[1, 0])),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, rt_col)
        if show_trans:
            drawRegion(src_img, trans_corners, trans_col, 1)
            cv2.putText(src_img, "Trans", (int(trans_corners[0, 0]), int(trans_corners[1, 0])),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, trans_col)
        if show_trans_mean:
            drawRegion(src_img, trans_mean_corners, trans_col, 1)
            cv2.putText(src_img, "Trans Mean", (int(trans_mean_corners[0, 0]), int(trans_mean_corners[1, 0])),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, trans_col)

        cv2.putText(src_img,
                    "h:{:7.4f} a:{:7.4f} s:{:7.4f} r:{:7.4f} t:{:7.4f} tm:{:7.4f}".format(
                        hom_error,
                        affine_error,
                        se2_error,
                        rt_error,
                        trans_error,
                        trans_mean_error),
                    (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

        switch_state_str = "{:7.4f}".format(curr_fps) + " ls: " + states[use_affine_ls] + "  inv: " + states[
            use_inverse_decomposition]
        cv2.putText(src_img, switch_state_str, (5, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
        cv2.imshow(window_name, src_img)



        if update_base_corners:
            base_corners_norm = np.copy(curr_corners_norm)
            base_corners_hm = util.homogenize(base_corners_norm)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('p'):
            cv2.waitKey(0)
        elif key == ord('h'):
            show_hom = 1 - show_hom
        elif key == ord('a'):
            show_affine = 1 - show_affine
        elif key == ord('s'):
            show_se2 = 1 - show_se2
        elif key == ord('r'):
            show_rt = 1 - show_rt
        elif key == ord('t'):
            show_trans = 1 - show_trans
        elif key == ord('m'):
            show_trans_mean = 1 - show_trans_mean
        elif key == ord('i'):
            use_inverse_decomposition = 1 - use_inverse_decomposition
        elif key == ord('l'):
            use_affine_ls = 1 - use_affine_ls
        elif key == ord('g'):
            use_jaccard_error = 1 - use_jaccard_error

    print '\n'
    avg_hom_error = np.mean(hom_errors)
    print 'avg_hom_error: ', avg_hom_error

    avg_affine_error = np.mean(affine_errors)
    print 'avg_affine_error: ', avg_affine_error

    # avg_affine_rec_error = np.mean(affine_rec_errors)
    # print 'avg_affine_rec_error: ', avg_affine_rec_error

    avg_se2_error = np.mean(se2_errors)
    print 'avg_se2_error: ', avg_se2_error

    avg_rt_error = np.mean(rt_errors)
    print 'avg_rt_error: ', avg_rt_error

    avg_trans_error = np.mean(trans_errors)
    print 'avg_trans_error: ', avg_trans_error

    avg_trans_mean_error = np.mean(trans_mean_errors)
    print 'avg_trans_mean_error: ', avg_trans_mean_error

    # avg_affine_fwd_error = np.mean(affine_fwd_errors)
    # avg_affine_inv_error = np.mean(affine_inv_errors)
    # avg_affine_mean_error = np.mean(affine_mean_errors)
    # avg_intra_affine_error = np.mean(intras_affine_errors)

    # print 'avg_affine_fwd_error: ', avg_affine_fwd_error
    # print 'avg_affine_inv_error: ', avg_affine_inv_error
    # print 'avg_affine_mean_error: ', avg_affine_mean_error
    # print 'avg_intra_affine_error: ', avg_intra_affine_error

    mean_fps = np.mean(error_fps)
    print 'mean_fps: ', mean_fps

    if use_affine_ls:
        if use_inverse_decomposition:
            result_fname = 'result_ls_inv.txt'
        else:
            result_fname = 'result_ls_fwd.txt'
    else:
        if use_inverse_decomposition:
            result_fname = 'result_inv.txt'
        else:
            result_fname = 'result_fwd.txt'
    result_file = open(result_fname, 'a')
    result_file.write('{:d}\t{:s}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{:9.6f}\n'.
                      format(seq_id, seq_name, avg_hom_error, avg_affine_error, avg_se2_error, avg_rt_error,
                             avg_trans_error, avg_trans_mean_error))
    result_file.close()

    print '*' * 100

    hom_file.close()
    affine_file.close()
    se2_file.close()
    rt_file.close()
    trans_file.close()














