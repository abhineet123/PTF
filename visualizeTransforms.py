from distanceGrid import *
import time
import os


def adjustCropLocation():
    global min_x, max_x, min_y, max_y
    min_x = crop_center_x - crop_size / 2
    max_x = crop_center_x + crop_size / 2
    min_y = crop_center_y - crop_size / 2
    max_y = crop_center_y + crop_size / 2

    if min_x < 0:
        min_x = 0
    if min_y < 0:
        min_y = 0
    if max_x >= img_width:
        max_x = img_width - 1
    if max_y >= img_height:
        max_y = img_height - 1


def mouseHandler(event, x, y, flags=None, param=None):
    global crop_center_x, crop_center_y, crop_size, crop_diff

    if event == cv2.EVENT_LBUTTONDOWN:
        crop_center_x = x
        crop_center_y = y
    elif event == cv2.EVENT_LBUTTONUP:
        pass
    elif event == cv2.EVENT_RBUTTONDOWN:
        crop_size += crop_diff
    elif event == cv2.EVENT_RBUTTONUP:
        pass
    elif event == cv2.EVENT_MBUTTONDOWN:
        crop_size -= crop_diff
    elif event == cv2.EVENT_MOUSEMOVE:
        pass
    adjustCropLocation()


def updateTrackbars():
    global tx, ty, theta, scale, a, b, v1, v2
    global tx_id, ty_id, theta_id, scale_id, a_id, b_id, v1_id, v2_id

    tx_id = int((tx - tx_min) * trackbar_res / (tx_max - tx_min))
    ty_id = int((ty - ty_min) * trackbar_res / (ty_max - ty_min))
    theta_id = int((theta - theta_min) * trackbar_res / (theta_max - theta_min))
    scale_id = int((scale - scale_min) * trackbar_res / (scale_max - scale_min))
    a_id = int((a - a_min) * trackbar_res / (a_max - a_min))
    b_id = int((b - b_min) * trackbar_res / (b_max - b_min))
    v1_id = int((v1 - v1_min) * trackbar_res / (v1_max - v1_min))
    v2_id = int((v2 - v2_min) * trackbar_res / (v2_max - v2_min))

    cv2.setTrackbarPos('tx', transform_win_name, tx_id)
    cv2.setTrackbarPos('ty', transform_win_name, ty_id)
    cv2.setTrackbarPos('theta', transform_win_name, theta_id)
    cv2.setTrackbarPos('scale', transform_win_name, scale_id)
    cv2.setTrackbarPos('a', transform_win_name, a_id)
    cv2.setTrackbarPos('b', transform_win_name, b_id)
    cv2.setTrackbarPos('v1', transform_win_name, v1_id)
    cv2.setTrackbarPos('v2', transform_win_name, v2_id)


def updateParams(x):
    global tx, ty, theta, scale, a, b, v1, v2
    global tx_id, ty_id, theta_id, scale_id, a_id, b_id, v1_id, v2_id

    tx_id = cv2.getTrackbarPos('tx', transform_win_name)
    ty_id = cv2.getTrackbarPos('ty', transform_win_name)
    theta_id = cv2.getTrackbarPos('theta', transform_win_name)
    scale_id = cv2.getTrackbarPos('scale', transform_win_name)
    a_id = cv2.getTrackbarPos('a', transform_win_name)
    b_id = cv2.getTrackbarPos('b', transform_win_name)
    v1_id = cv2.getTrackbarPos('v1', transform_win_name)
    v2_id = cv2.getTrackbarPos('v2', transform_win_name)

    tx = float(float(tx_id * (tx_max - tx_min)) / trackbar_res + tx_min)
    ty = float(float(ty_id * (ty_max - ty_min)) / trackbar_res + ty_min)
    theta = float(float(theta_id * (theta_max - theta_min)) / trackbar_res + theta_min)
    scale = float(float(scale_id * (scale_max - scale_min)) / trackbar_res + scale_min)
    a = float(float(a_id * (a_max - a_min)) / trackbar_res + a_min)
    b = float(float(b_id * (b_max - b_min)) / trackbar_res + b_min)
    v1 = float(float(v1_id * (v1_max - v1_min)) / trackbar_res + v1_min)
    v2 = float(float(v2_id * (v2_max - v2_min)) / trackbar_res + v2_min)

    # print 'updating params...'
    # print 'tx: ', tx
    # print 'ty: ', ty
    # print 'theta: ', theta
    # print 'scale: ', scale
    # print 'a: ', a
    # print 'b: ', b
    # print 'v1: ', v1
    # print 'v2 ', v2

    updateCornersPost()
    updateCornersPre()
    printCornerErrors()

    drawCorners()


def drawCorners():
    global pre_disp_img, post_disp_img, curr_img, transform_vals_img

    # print 'drawing corners...'

    pre_disp_img = curr_img.copy()
    post_disp_img = curr_img.copy()
    param_text = ''

    if show_gt:
        drawRegion(pre_disp_img, curr_corners, gt_col, 1)
        cv2.putText(pre_disp_img, "GT", (int(curr_corners[0, 0]), int(curr_corners[1, 0])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, gt_col)
        drawRegion(post_disp_img, curr_corners, gt_col, 1)
        cv2.putText(post_disp_img, "GT", (int(curr_corners[0, 0]), int(curr_corners[1, 0])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, gt_col)

    if show_trans:
        drawRegion(pre_disp_img, pre_trans_corners, trans_col, 1)
        cv2.putText(pre_disp_img, "Trans", (int(pre_trans_corners[0, 0]), int(pre_trans_corners[1, 0])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, trans_col)
        drawRegion(post_disp_img, post_trans_corners, trans_col, 1)
        cv2.putText(post_disp_img, "Trans", (int(post_trans_corners[0, 0]), int(post_trans_corners[1, 0])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, trans_col)
        param_text += 'tx: {:12.9f} '.format(tx)
        param_text += 'ty: {:12.9f} '.format(ty)

    if show_rs:
        drawRegion(pre_disp_img, pre_rs_corners, rs_col, 1)
        cv2.putText(pre_disp_img, "RS", (int(pre_rs_corners[0, 0]), int(pre_rs_corners[1, 0])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, rs_col)
        drawRegion(post_disp_img, post_rs_corners, rs_col, 1)
        cv2.putText(post_disp_img, "RS", (int(post_rs_corners[0, 0]), int(post_rs_corners[1, 0])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, rs_col)
        param_text += 'theta: {:12.9f} '.format(theta)
        param_text += 'scale: {:12.9f} '.format(scale)

    if show_shear:
        drawRegion(pre_disp_img, pre_shear_corners, shear_col, 1)
        cv2.putText(pre_disp_img, "Shear", (int(pre_shear_corners[0, 0]), int(pre_shear_corners[1, 0])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, shear_col)
        drawRegion(post_disp_img, post_shear_corners, shear_col, 1)
        cv2.putText(post_disp_img, "Shear", (int(post_shear_corners[0, 0]), int(post_shear_corners[1, 0])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, shear_col)
        param_text += 'a: {:12.9f} '.format(a)
        param_text += 'b: {:12.9f} '.format(b)

    if show_proj:
        drawRegion(pre_disp_img, pre_proj_corners, proj_col, 1)
        cv2.putText(pre_disp_img, "Proj", (int(pre_proj_corners[0, 0]), int(pre_proj_corners[1, 0])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, proj_col)
        drawRegion(post_disp_img, post_proj_corners, proj_col, 1)
        cv2.putText(post_disp_img, "Proj", (int(post_proj_corners[0, 0]), int(post_proj_corners[1, 0])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, proj_col)
        param_text += 'v1: {:12.9f} '.format(v1)
        param_text += 'v2: {:12.9f} '.format(v2)

    transform_vals_img = np.zeros((250, 500, 3), dtype=np.uint8)
    cv2.putText(transform_vals_img, 'tx: {:12.9f} '.format(tx), (5, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.00,
                (255, 255, 255))
    cv2.putText(transform_vals_img, 'ty: {:12.9f} '.format(ty), (5, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.00,
                (255, 255, 255))
    cv2.putText(transform_vals_img, 'theta: {:12.9f} '.format(theta), (5, 85), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.00,
                (255, 255, 255))
    cv2.putText(transform_vals_img, 'scale: {:12.9f} '.format(scale), (5, 115), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.00,
                (255, 255, 255))
    cv2.putText(transform_vals_img, 'a: {:12.9f} '.format(a), (5, 145), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.00,
                (255, 255, 255))
    cv2.putText(transform_vals_img, 'b: {:12.9f} '.format(b), (5, 175), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.00,
                (255, 255, 255))
    cv2.putText(transform_vals_img, 'v1: {:12.9f} '.format(v1), (5, 205), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.00,
                (255, 255, 255))
    cv2.putText(transform_vals_img, 'v2: {:12.9f} '.format(v2), (5, 235), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.00,
                (255, 255, 255))

    cv2.putText(pre_disp_img, param_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (255, 255, 255))
    cv2.putText(post_disp_img, param_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (255, 255, 255))


def updateCornersPost():
    global tx, ty, theta, scale, a, b, v1, v2
    global post_trans_mat, post_rot_mat, post_scale_mat, post_rs_mat, post_proj_mat
    global post_trans_corners, post_rs_corners, post_shear_corners, post_proj_corners

    curr_norm_hom_mat = curr_norm_mat * curr_hom_mat

    post_trans_mat = getTranslationMatrix(tx, ty)
    pw_mat = post_trans_mat
    post_trans_corners = util.dehomogenize(curr_norm_hom_mat * pw_mat * std_corners_hm)

    post_rot_mat = getRotationMatrix(theta)
    post_scale_mat = getScalingMatrix(scale)
    pw_mat = pw_mat * post_scale_mat * post_rot_mat
    post_rs_corners = util.dehomogenize(curr_norm_hom_mat * pw_mat * std_corners_hm)

    post_shear_mat = getShearingMatrix(a, b)
    pw_mat = pw_mat * post_shear_mat
    post_shear_corners = util.dehomogenize(curr_norm_hom_mat * pw_mat * std_corners_hm)

    post_proj_mat = getProjectionMatrix(v1, v2)
    pw_mat = pw_mat * post_proj_mat
    post_proj_corners = util.dehomogenize(curr_norm_hom_mat * pw_mat * std_corners_hm)


def updateCornersPre():
    global tx, ty, theta, scale, a, b, v1, v2
    global pre_trans_mat, pre_rot_mat, pre_scale_mat, pre_shear_mat, pre_proj_mat
    global pre_trans_corners, pre_rs_corners, pre_shear_corners, pre_proj_corners

    # print 'updating corners...'

    curr_norm_hom_mat = curr_norm_mat * curr_hom_mat

    pre_trans_mat = getTranslationMatrix(tx, ty)
    pw_mat = pre_trans_mat
    pre_trans_corners = util.dehomogenize(curr_norm_hom_mat * pw_mat * std_corners_hm)

    pre_rot_mat = getRotationMatrix(theta)
    pre_scale_mat = getScalingMatrix(scale)
    pw_mat = pre_scale_mat * pre_rot_mat * pw_mat
    pre_rs_corners = util.dehomogenize(curr_norm_hom_mat * pw_mat * std_corners_hm)

    pre_shear_mat = getShearingMatrix(a, b)
    pw_mat = pre_shear_mat * pw_mat
    pre_shear_corners = util.dehomogenize(curr_norm_hom_mat * pw_mat * std_corners_hm)

    pre_proj_mat = getProjectionMatrix(v1, v2)
    pw_mat = pre_proj_mat * pw_mat
    pre_proj_corners = util.dehomogenize(curr_norm_hom_mat * pw_mat * std_corners_hm)


def printCornerErrors():
    global pre_trans_corners, pre_rs_corners, pre_shear_corners, pre_proj_corners
    global post_trans_corners, post_rs_corners, post_shear_corners, post_proj_corners

    trans_error = math.sqrt(np.sum(np.square(pre_trans_corners - post_trans_corners)) / 4)
    rs_error = math.sqrt(np.sum(np.square(pre_rs_corners - post_rs_corners)) / 4)
    shear_error = math.sqrt(np.sum(np.square(pre_shear_corners - post_shear_corners)) / 4)
    proj_error = math.sqrt(np.sum(np.square(pre_proj_corners - post_proj_corners)) / 4)

    print 'trans_error: {:15.12f} rs_error: {:15.12f} shear_error: {:15.12f} proj_error: {:15.12f}'.format(trans_error,
                                                                                                           rs_error,
                                                                                                           shear_error,
                                                                                                           proj_error)
    # print 'rs_error: ', rs_error
    # print 'shear_error: ', shear_error
    # print 'proj_error: ', proj_error


if __name__ == '__main__':

    grid_types = {0: 'trans',
                  1: 'rs',
                  2: 'shear',
                  3: 'proj',
                  4: 'rtx',
                  5: 'rty',
                  6: 'stx',
                  7: 'sty'
    }
    filter_types = {0: 'none',
                    1: 'gauss',
                    2: 'box',
                    3: 'norm_box',
                    4: 'bilateral',
                    5: 'median',
                    6: 'gabor',
                    7: 'sobel',
                    8: 'scharr',
                    9: 'LoG',
                    10: 'DoG',
                    11: 'laplacian',
                    12: 'canny'
    }
    sequences = {0: 'nl_bookI_s3',
                 1: 'nl_bookII_s3',
                 2: 'nl_bookIII_s3',
                 3: 'nl_cereal_s3',
                 4: 'nl_juice_s3',
                 5: 'nl_mugI_s3',
                 6: 'nl_mugII_s3',
                 7: 'nl_mugIII_s3',
                 8: 'nl_bookI_s4',
                 9: 'nl_bookII_s4',
                 10: 'nl_bookIII_s4',
                 11: 'nl_cereal_s4',
                 12: 'nl_juice_s4',
                 13: 'nl_mugI_s4',
                 14: 'nl_mugII_s4',
                 15: 'nl_mugIII_s4',
                 16: 'nl_bus',
                 17: 'nl_highlighting',
                 18: 'nl_letter',
                 19: 'nl_newspaper',
    }
    db_root_path = '/home/abhineet/E/UofA/Thesis/Code/Datasets'
    actor = 'Human'
    seq_id = 3
    inc_type = 'ic'
    grid_id = 1
    filter_id = 0
    kernel_size = 10
    start_id = 1
    write_img_data = 0

    trackbar_res = 10000
    crop_size = 100
    crop_mag_size = 500
    crop_diff = 10

    gt_col = (0, 0, 0)
    trans_col = (255, 255, 255)
    rs_col = (0, 0, 255)
    shear_col = (0, 255, 0)
    proj_col = (255, 255, 0)

    show_gt = 0
    show_trans = 1
    show_rs = 1
    show_shear = 1
    show_proj = 1

    update_func = updateCornersPost

    std_resx = 50
    std_resy = 50
    n_pts = std_resx * std_resy

    tx_res, ty_res = [80, 80]
    theta_res, scale_res = [100, 60]
    a_res, b_res = [60, 80]
    v1_res, v2_res = [60, 80]

    trans_thr = 1
    theta_thresh, scale_thresh = [np.pi / 4, 0.5]
    a_thresh, b_thresh = [1, 1]
    v1_thresh, v2_thresh = [1, 1]

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

    tx_vec = np.insert(tx_vec, np.argwhere(tx_vec >= 0)[0, 0], 0)
    ty_vec = np.insert(ty_vec, np.argwhere(ty_vec >= 0)[0, 0], 0)
    theta_vec = np.insert(theta_vec, np.argwhere(theta_vec >= 0)[0, 0], 0)
    scale_vec = np.insert(scale_vec, np.argwhere(scale_vec >= 0)[0, 0], 0)
    a_vec = np.insert(a_vec, np.argwhere(a_vec >= 0)[0, 0], 0)
    b_vec = np.insert(b_vec, np.argwhere(b_vec >= 0)[0, 0], 0)
    v1_vec = np.insert(v1_vec, np.argwhere(v1_vec >= 0)[0, 0], 0)
    v2_vec = np.insert(v2_vec, np.argwhere(v2_vec >= 0)[0, 0], 0)

    arg_id = 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        grid_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        filter_id = int(sys.argv[arg_id])
        arg_id += 1

    if seq_id >= len(sequences):
        print 'Invalid dataset_id: ', seq_id
        sys.exit()
    if grid_id >= len(grid_types):
        print 'Invalid grid_id: ', grid_id
        sys.exit()
    if filter_id >= len(filter_types):
        print 'Invalid filter_id: ', filter_id
        sys.exit()

    seq_name = sequences[seq_id]
    grid_type = grid_types[grid_id]
    filter_type = filter_types[filter_id]

    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    print 'inc_type: ', inc_type
    print 'grid_type: ', grid_type
    print 'filter_type: ', filter_type
    print 'kernel_size: ', kernel_size

    src_folder = db_root_path + '/' + actor + '/' + seq_name

    if filter_type != 'none':
        img_folder = 'Image Data/' + seq_name + '_' + filter_type + str(kernel_size)
        dist_folder = 'Distance Data/' + seq_name + '_' + filter_type + str(
            kernel_size) + '/' + inc_type + '_' + grid_type
    else:
        img_folder = 'Image Data/' + seq_name
        dist_folder = 'Distance Data/' + seq_name + '/' + inc_type + '_' + grid_type

    if not os.path.exists(dist_folder):
        os.makedirs(dist_folder)

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    ground_truth_fname = db_root_path + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)
    no_of_frames = ground_truth.shape[0]
    print 'no_of_frames: ', no_of_frames

    end_id = no_of_frames

    init_corners = np.asarray([ground_truth[0, 0:2].tolist(),
                               ground_truth[0, 2:4].tolist(),
                               ground_truth[0, 4:6].tolist(),
                               ground_truth[0, 6:8].tolist()]).T

    std_pts, std_corners = getNormalizedUnitSquarePts(std_resx, std_resy)
    std_corners_hm = util.homogenize(std_corners)
    std_pts_hm = util.homogenize(std_pts)
    (init_corners_norm, init_norm_mat) = getNormalizedPoints(init_corners)
    init_hom_mat = np.mat(util.compute_homography(std_corners, init_corners_norm))
    init_pts_norm = util.dehomogenize(init_hom_mat * std_pts_hm)
    init_pts = util.dehomogenize(init_norm_mat * util.homogenize(init_pts_norm))

    init_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(1))

    init_img_gs = cv2.cvtColor(init_img, cv2.cv.CV_BGR2GRAY).astype(np.float64)
    img_height, img_width = init_img_gs.shape
    if filter_type != 'none':
        init_img_gs = applyFilter(init_img_gs, filter_type)

    if write_img_data:
        init_img_gs.astype(np.uint8).tofile(img_folder + '/' + 'frame_0_gs.bin')
    # k=util.bilin_interp(init_img_gs, init_pts[0, 0], init_pts[1, 0])
    # print 'k: ', k
    init_pixel_vals = np.mat([util.bilin_interp(init_img_gs, init_pts[0, pt_id], init_pts[1, pt_id]) for pt_id in
                              xrange(n_pts)])


    # cv2.imshow('Init Image', init_img)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    np.savetxt(dist_folder + '/init_pixel_vals.txt', init_pixel_vals)

    # if grid_type == 'trans':
    # np.savetxt(dist_folder + '/tx_vec.txt', tx_vec)
    # np.savetxt(dist_folder + '/ty_vec.txt', ty_vec)
    # print 'tx_vec: ', tx_vec
    # print 'ty_vec: ', ty_vec
    # elif grid_type == 'rtx':
    # np.savetxt(dist_folder + '/tx_vec.txt', tx_vec)
    # np.savetxt(dist_folder + '/theta_vec.txt', theta_vec)
    # print 'tx_vec: ', tx_vec
    # print 'theta_vec: ', theta_vec
    # elif grid_type == 'rty':
    # np.savetxt(dist_folder + '/ty_vec.txt', ty_vec)
    # np.savetxt(dist_folder + '/theta_vec.txt', theta_vec)
    #     print 'ty_vec: ', ty_vec
    #     print 'theta_vec: ', theta_vec
    # elif grid_type == 'rs':
    #     np.savetxt(dist_folder + '/scale_vec.txt', scale_vec)
    #     np.savetxt(dist_folder + '/theta_vec.txt', theta_vec)
    #     print 'scale_vec: ', scale_vec
    #     print 'theta_vec: ', theta_vec
    # elif grid_type == 'shear':
    #     np.savetxt(dist_folder + '/a_vec.txt', a_vec)
    #     np.savetxt(dist_folder + '/b_vec.txt', b_vec)
    #     print 'a_vec: ', a_vec
    #     print 'b_vec: ', b_vec
    # elif grid_type == 'proj':
    #     np.savetxt(dist_folder + '/v1_vec.txt', v1_vec)
    #     np.savetxt(dist_folder + '/v2_vec.txt', v2_vec)
    #     print 'v1_vec: ', v1_vec
    #     print 'v2_vec: ', v2_vec
    # else:
    #     raise StandardError('Invalid grid_type: ' + grid_type)

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

    crop_center_x = np.mean(curr_corners[0, :])
    crop_center_y = np.mean(curr_corners[1, :])
    adjustCropLocation()

    pre_img_win_name = 'Pre Transform Image'
    post_img_win_name = 'Post Transform Image'
    transform_win_name = 'Transform Values'
    pre_cropped_img_win_name = 'Cropped Pre Transform Image'
    post_cropped_img_win_name = 'Cropped Post Transform Image'

    cv2.namedWindow(pre_img_win_name)
    cv2.namedWindow(post_img_win_name)
    cv2.namedWindow(pre_cropped_img_win_name)
    cv2.namedWindow(post_cropped_img_win_name)
    cv2.namedWindow(transform_win_name, cv2.cv.CV_WINDOW_NORMAL)

    cv2.setMouseCallback(pre_img_win_name, mouseHandler)
    cv2.setMouseCallback(post_img_win_name, mouseHandler)
    cv2.setMouseCallback(pre_cropped_img_win_name, mouseHandler)
    cv2.setMouseCallback(post_cropped_img_win_name, mouseHandler)

    # cv2.imshow(transform_win_name, init_img)
    pre_disp_img = init_img.copy()
    post_disp_img = init_img.copy()

    transform_vals_img = np.zeros((250, 500), dtype=np.uint8)

    tx, ty, theta, scale, a, b, v1, v2 = [0] * 8
    tx_id, ty_id, theta_id, scale_id, a_id, b_id, v1_id, v2_id = [trackbar_res / 2] * 8

    cv2.createTrackbar('tx', transform_win_name, tx_id, trackbar_res, updateParams)
    cv2.createTrackbar('ty', transform_win_name, ty_id, trackbar_res, updateParams)
    cv2.createTrackbar('theta', transform_win_name, theta_id, trackbar_res, updateParams)
    cv2.createTrackbar('scale', transform_win_name, scale_id, trackbar_res, updateParams)
    cv2.createTrackbar('a', transform_win_name, a_id, trackbar_res, updateParams)
    cv2.createTrackbar('b', transform_win_name, b_id, trackbar_res, updateParams)
    cv2.createTrackbar('v1', transform_win_name, v1_id, trackbar_res, updateParams)
    cv2.createTrackbar('v2', transform_win_name, v2_id, trackbar_res, updateParams)

    curr_img = init_img

    updateTrackbars()

    pause_seq = 0
    frame_id = start_id

    updateCornersPost()
    updateCornersPre()

    while frame_id <= end_id:
        if not pause_seq:
            pause_seq = 1

            print 'frame_id: ', frame_id

            # ret, curr_img = cap.read()
            curr_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id + 1))


            # print 'curr_img: ', curr_img
            if curr_img is None:
                break
            curr_img_gs = cv2.cvtColor(curr_img, cv2.cv.CV_BGR2GRAY).astype(np.float64)
            if filter_type != 'none':
                curr_img_gs = applyFilter(curr_img_gs, filter_type)

            curr_pixel_vals = np.mat(
                [util.bilin_interp(curr_img_gs, curr_pts[0, pt_id], curr_pts[1, pt_id]) for pt_id in
                 xrange(n_pts)])

            curr_corners = np.asarray([ground_truth[frame_id, 0:2].tolist(),
                                       ground_truth[frame_id, 2:4].tolist(),
                                       ground_truth[frame_id, 4:6].tolist(),
                                       ground_truth[frame_id, 6:8].tolist()]).T

            curr_corners_norm, curr_norm_mat = getNormalizedPoints(curr_corners)
            curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners_norm))

            # crop_center_x = np.mean(curr_corners[0, :])
            # crop_center_y = np.mean(curr_corners[1, :])
            # tx, ty, theta, scale, a, b, v1, v2 = getHomographyParamsInverse(curr_hom_mat)

            curr_pts_norm = util.dehomogenize(curr_hom_mat * std_pts_hm)
            curr_pts = util.dehomogenize(curr_norm_mat * util.homogenize(curr_pts_norm))

            frame_id += 1

        drawCorners()

        pre_cropped_img = pre_disp_img[min_y:max_y, min_x:max_x]
        pre_cropped_img = cv2.resize(pre_cropped_img, (crop_mag_size, crop_mag_size))

        post_cropped_img = post_disp_img[min_y:max_y, min_x:max_x]
        post_cropped_img = cv2.resize(post_cropped_img, (crop_mag_size, crop_mag_size))

        cv2.imshow(pre_img_win_name, pre_disp_img)
        cv2.imshow(post_img_win_name, post_disp_img)
        cv2.imshow(pre_cropped_img_win_name, pre_cropped_img)
        cv2.imshow(post_cropped_img_win_name, post_cropped_img)
        cv2.imshow(transform_win_name, transform_vals_img)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            pause_seq = 1 - pause_seq
        elif key == ord('t'):
            show_trans = 1 - show_trans
        elif key == ord('r'):
            show_rs = 1 - show_rs
        elif key == ord('s'):
            show_shear = 1 - show_shear
        elif key == ord('p'):
            show_proj = 1 - show_proj
        elif key == ord('g'):
            show_gt = 1 - show_gt
        elif key == ord('m'):
            tx, ty, theta, scale, a, b, v1, v2 = [0] * 8
            updateTrackbars()





