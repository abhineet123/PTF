from Misc import readTrackingData
from Misc import getParamDict
from Misc import readDistGridParams
from Misc import drawRegion
from decomposition import *

import sys
import cv2
import numpy as np
import csv

if __name__ == '__main__':
    db_root_dir = '../Datasets/TrakMark'
    seq_type = 'Film Studio'
    seq_name = 'rits02_00'
    img_name_fmt = 'frame%05d.jpg'
    gt_col = (0, 255, 0)
    pause_seq = 1

    src_dir = '{:s}/{:s}/{:s}'.format(db_root_dir, seq_type, seq_name)
    gt_param_fname = '{:s}/{:s}/{:s}_IS900.csv'.format(db_root_dir, seq_type, seq_name)
    gt_aist_param_fname = '{:s}/{:s}/{:s}_IS900_AIST.csv'.format(db_root_dir, seq_type, seq_name)
    gt_fname = '{:s}/{:s}/{:s}.txt'.format(db_root_dir, seq_type, seq_name)

    print 'Reading ground truth parameters from: ', gt_param_fname
    src_fname = '{:s}/{:s}'.format(src_dir, img_name_fmt)
    enable_aist = 0
    if os.path.isfile(gt_aist_param_fname):
        enable_aist = 1

    float_formatter = lambda x: "%12.6f" % x
    np.set_printoptions(formatter={'float_kind': float_formatter})

    if enable_aist:
        intrinsic_mat = np.mat(
            [[6.0816190294271473e+002, 0., 3.5772820637387372e+002],
             [0, 6.0801900732727700e+002, 2.0751923371619617e+002],
             [0, 0, 1]],
            dtype=np.float64)
    else:
        SCRX = 720.000000
        SCRY = 405.000000
        CCDX = 9.589680
        CCDY = 5.390145
        SX = 1.005812
        CAMF = 7.000474
        KAPPA1 = 0.004722
        KAPPA2 = -0.000111
        IMAGE_CENTER_X = 1.853480
        IMAGE_CENTER_Y = -3.175373



        mx = SCRX / CCDX
        my = SCRY / CCDY

        alphax = CAMF * mx
        alphay = CAMF * my

        cx = IMAGE_CENTER_X * 1
        cy = IMAGE_CENTER_Y * 1

        intrinsic_mat = np.asmatrix(np.zeros((3, 3), dtype=np.float64))
        intrinsic_mat[0, 0] = alphax
        intrinsic_mat[0, 1] = 0
        intrinsic_mat[0, 2] = cx
        intrinsic_mat[1, 1] = alphay
        intrinsic_mat[1, 2] = cy
        intrinsic_mat[2, 2] = 1

    print 'intrinsic_mat: \n', intrinsic_mat

    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        print 'The source file ', src_fname, ' could not be opened'
        sys.exit()
    ret, init_img = cap.read()

    if os.path.isfile(gt_fname):
        ground_truth = readTrackingData(gt_fname)
        init_corners = np.asarray([ground_truth[0, 0:2].tolist(),
                                   ground_truth[0, 2:4].tolist(),
                                   ground_truth[0, 4:6].tolist(),
                                   ground_truth[0, 6:8].tolist()]).T
    else:
        init_corners = getTrackingObject2(init_img, col=(0, 0, 255), title='Select initial object location')
        init_corners = np.asmatrix(init_corners).T
        out_file = open(gt_fname, 'w')
        out_file.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
        writeCorners(out_file, init_corners, 1)
        out_file.close()

    init_corners_hm = util.homogenize(init_corners)

    gt_corners_window_name = 'Ground Truth Corners'
    cv2.namedWindow(gt_corners_window_name)
    cv2.imshow(gt_corners_window_name, init_img)

    img_height = init_img.shape[0]
    img_width = init_img.shape[1]

    print 'img_height: ', img_height
    print 'img_width: ', img_width

    y_axis_trans_mat = np.asmatrix(np.identity(3, dtype=np.float64))
    y_axis_trans_mat[1, 2] = -img_height
    # y_axis_trans_mat[0, 2] = img_width
    y_axis_refl_mat = np.asmatrix(np.identity(3, dtype=np.float64))
    y_axis_refl_mat[1, 1] = -1

    intrinsic_mat = y_axis_trans_mat * y_axis_refl_mat * intrinsic_mat

    print 'intrinsic_mat with axis flipping: \n', intrinsic_mat

    rotate_x_mat = np.asmatrix(np.identity(3, dtype=np.float64))
    rotate_y_mat = np.asmatrix(np.identity(3, dtype=np.float64))
    rotate_z_mat = np.asmatrix(np.identity(3, dtype=np.float64))
    trans_mat = np.asmatrix(np.zeros((3, 1), dtype=np.float64))

    extrinsic_mat = np.asmatrix(np.zeros((3, 4), dtype=np.float64))
    extrinsic_mat_inv = np.asmatrix(np.zeros((3, 4), dtype=np.float64))

    curr_corners = init_corners.copy()
    curr_img = init_img

    if enable_aist:
        aist_file = open(gt_aist_param_fname, 'rb')
        aist_gt_reader = csv.DictReader(aist_file, delimiter=',')
        # aist_file.close()

    init_corners3d_hm = None
    with open(gt_param_fname, 'rb') as gt_param_file:
        gt_reader = csv.DictReader(gt_param_file, delimiter=',')
        frame_id = 1
        for row, row2 in zip(gt_reader, aist_gt_reader):
            print '*' * 40, 'frame : ', frame_id, '*' * 40
            roll = float(row['Rotate X'])
            pitch = float(row['Rotate Y'])
            yaw = float(row['Rotate Z'])

            cos_theta = math.cos(math.radians(roll))
            sin_theta = math.sin(math.radians(roll))

            rotate_x_mat[1, 1] = cos_theta
            rotate_x_mat[1, 2] = -sin_theta
            rotate_x_mat[2, 1] = sin_theta
            rotate_x_mat[2, 2] = cos_theta

            cos_theta = math.cos(math.radians(pitch))
            sin_theta = math.sin(math.radians(pitch))

            rotate_y_mat[0, 0] = cos_theta
            rotate_y_mat[2, 0] = -sin_theta
            rotate_y_mat[0, 2] = sin_theta
            rotate_y_mat[2, 2] = cos_theta

            cos_theta = math.cos(math.radians(yaw))
            sin_theta = math.sin(math.radians(yaw))

            rotate_z_mat[0, 0] = cos_theta
            rotate_z_mat[0, 1] = -sin_theta
            rotate_z_mat[1, 0] = sin_theta
            rotate_z_mat[1, 1] = cos_theta

            rotate_mat = rotate_z_mat * rotate_y_mat * rotate_x_mat
            rotate_mat_inv = rotate_mat.transpose()

            tx = row['Position X']
            ty = row['Position Y']
            tz = row['Position Z']
            trans_mat[0] = tx
            trans_mat[1] = ty
            trans_mat[2] = tz

            trans_mat_inv = - rotate_mat_inv * trans_mat

            extrinsic_mat[:, 0:3] = rotate_mat
            extrinsic_mat[:, 3] = trans_mat

            extrinsic_mat_inv[:, 0:3] = rotate_mat_inv
            extrinsic_mat_inv[:, 3] = trans_mat_inv

            print 'roll: ', roll
            print 'pitch: ', pitch
            print 'yaw: ', yaw
            print 'rotate_mat: \n', rotate_mat
            rotate_prod = rotate_mat * rotate_mat_inv
            print 'rotate_prod: \n', rotate_prod
            print 'trans_mat: \n', trans_mat
            print 'trans_mat_inv: \n', trans_mat_inv
            print 'tx: ', tx
            print 'ty: ', ty
            print 'tz: ', tz
            print 'extrinsic_mat: \n', extrinsic_mat
            print 'extrinsic_mat_inv: \n', extrinsic_mat_inv

            if enable_aist:
                m11 = float(row2['m11'])
                m12 = float(row2['m12'])
                m13 = float(row2['m13'])
                m14 = float(row2['m14'])

                m21 = float(row2['m21'])
                m22 = float(row2['m22'])
                m23 = float(row2['m23'])
                m24 = float(row2['m24'])

                m31 = float(row2['m31'])
                m32 = float(row2['m32'])
                m33 = float(row2['m33'])
                m34 = float(row2['m34'])

                aist_extrinsic_mat = np.mat([
                    [m11, m12, m13, m14],
                    [m21, m22, m23, m24],
                    [m31, m32, m33, m34]
                ])
                print 'aist_extrinsic_mat: \n', aist_extrinsic_mat
                extrinsic_mat = aist_extrinsic_mat.copy()

            transform_mat = intrinsic_mat * extrinsic_mat
            print 'transform_mat: \n', transform_mat
            if init_corners3d_hm is None:
                transform_mat_inv = np.linalg.pinv(transform_mat)
                # transform_mat_inv = np.linalg.inv(transform_mat.transpose() * transform_mat) * transform_mat.transpose()
                left_prod = transform_mat_inv * transform_mat
                right_prod = transform_mat * transform_mat_inv
                print 'transform_mat_inv: \n', transform_mat_inv
                print 'left_prod: \n', left_prod
                print 'right_prod: \n', right_prod
                print 'init_corners_hm: \n', init_corners_hm
                init_corners3d_hm = transform_mat_inv * init_corners_hm
                init_corners3d = util.dehomogenize(init_corners3d_hm)
                print 'init_corners3d_hm: \n', init_corners3d_hm
                print 'init_corners3d: \n', init_corners3d
            else:
                prev_corners = np.copy(curr_corners)
                curr_corners_hm = transform_mat * init_corners3d_hm
                curr_corners = util.dehomogenize(curr_corners_hm)
                # curr_corners[0, :]=img_width- curr_corners[0, :]
                # curr_corners[1, :]=img_height- curr_corners[1, :]
                corner_change = curr_corners - prev_corners
                # curr_corners = prev_corners - corner_change

                print 'curr_corners: \n', curr_corners
                print 'corner_change: \n', corner_change
                # for i in xrange(4):
                # x = curr_corners[0, i]
                # y = curr_corners[1, i]
                # r2 = x * x + y * y
                # r4 = r2 * r2
                # curr_corners[0, i] *= (1 + KAPPA1 * r2 + KAPPA2 * r4)
                # curr_corners[1, i] *= (1 + KAPPA1 * r2 + KAPPA2 * r4)
                # print 'curr_corners with distortion: \n', curr_corners

            if frame_id > 1:
                ret, curr_img = cap.read()
                if not ret:
                    print 'End of sequence reached unexpectedly'
                    break
            drawRegion(curr_img, curr_corners, gt_col, 1)
            cv2.imshow(gt_corners_window_name, curr_img)
            key = cv2.waitKey(1 - pause_seq)
            if key == 27:
                break
            elif key == 32:
                pause_seq = 1 - pause_seq

            frame_id += 1













