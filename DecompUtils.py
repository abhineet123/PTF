import numpy as np
import utility as util
import os
import cv2
import sys
import math
from Misc import *


def getRSMatrix(scale, theta):
    rot_mat = getRotationMatrix(theta)
    scale_mat = getScalingMatrix(scale)
    rs_mat = scale_mat * rot_mat
    return rs_mat


def getRotationMatrix(theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rot_mat = np.mat(
        [[cos_theta, - sin_theta, 0],
         [sin_theta, cos_theta, 0],
         [0, 0, 1]]
    )
    return rot_mat


def getTranslationMatrix(tx, ty):
    trans_mat = np.mat(
        [[1, 0, tx],
         [0, 1, ty],
         [0, 0, 1]]
    )
    return trans_mat


def getTranscalingMatrix(tx, ty, s):
    trans_mat = np.mat(
        [[1 + s, 0, tx],
         [0, 1 + s, ty],
         [0, 0, 1]]
    )
    return trans_mat


def getScalingMatrix(scale):
    scale_mat = np.mat(
        [[1 + scale, 0, 0],
         [0, 1 + scale, 0],
         [0, 0, 1]]
    )
    return scale_mat


def getShearingMatrix(a, b):
    shear_mat = np.mat(
        [[1 + a, b, 0],
         [0, 1, 0],
         [0, 0, 1]]
    )
    return shear_mat


def getProjectionMatrix(v1, v2):
    proj_mat = np.mat(
        [[1, 0, 0],
         [0, 1, 0],
         [v1, v2, 1]]
    )
    return proj_mat


def getJacobianTrans(params, in_pts, out_pts, pts_size):
    trans_mat = getTranslationMatrix(params[0], params[1])
    rec_pts = util.dehomogenize(trans_mat * in_pts)
    out_pts = np.mat(util.dehomogenize(out_pts))
    in_pts = np.mat(util.dehomogenize(in_pts))

    pts_diff = np.mat(rec_pts - out_pts).transpose()
    dtx = np.sum(pts_diff[:, 0])
    dty = np.sum(pts_diff[:, 1])

    jacobian = 2 * np.array([dtx, dty]) / pts_size
    return jacobian


def getJacobianRotate(theta, in_pts, out_pts, pts_size):
    rot_mat = getRotationMatrix(theta)
    rec_pts = util.dehomogenize(rot_mat * in_pts)
    out_pts = np.mat(util.dehomogenize(out_pts))
    in_pts = np.mat(util.dehomogenize(in_pts))
    pts_diff = np.mat(rec_pts - out_pts).transpose()
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    dtheta_mat = np.mat([[-sin_theta, -cos_theta],
                         [cos_theta, -sin_theta]])
    dtheta = 2 * np.dot(np.ravel(pts_diff), np.ravel(dtheta_mat * in_pts, order='F')) / pts_size
    return dtheta


def getJacobianScale(scale, in_pts, out_pts, pts_size):
    scale_mat = getScalingMatrix(scale)
    rec_pts = util.dehomogenize(scale_mat * in_pts)
    out_pts = np.mat(util.dehomogenize(out_pts))
    in_pts = np.mat(util.dehomogenize(in_pts))
    pts_diff = np.mat(rec_pts - out_pts).transpose()
    ds = 2 * np.dot(np.ravel(pts_diff), np.ravel(in_pts, order='F')) / pts_size
    return ds


def getJacobianShear(params, in_pts, out_pts, pts_size):
    shear_mat = getShearingMatrix(params[0], params[1])
    rec_pts = shear_mat * in_pts
    out_pts = np.mat(util.dehomogenize(out_pts))
    in_pts = np.mat(util.dehomogenize(in_pts))

    pts_diff = np.mat(rec_pts - out_pts).transpose()
    da = np.dot(pts_diff[0, :], in_pts[0, :])
    db = np.dot(pts_diff[0, :], in_pts[1, :])
    jacobian = 2 * np.array([da, db]) / pts_size
    return jacobian


def getJacobianRT(params, in_pts, out_pts, pts_size):
    (trans_mat, rot_mat) = getDecomposedRTMatrices(params)
    rec_pts = util.dehomogenize(trans_mat * rot_mat * in_pts)
    out_pts = np.mat(util.dehomogenize(out_pts))
    in_pts = np.mat(util.dehomogenize(in_pts))

    pts_diff = np.mat(rec_pts - out_pts).transpose()
    theta = params[2]
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    dtheta_mat = np.mat([[-sin_theta, -cos_theta],
                         [cos_theta, -sin_theta]])

    dtheta = np.dot(np.ravel(pts_diff), np.ravel(dtheta_mat * in_pts, order='F'))
    dtx = np.sum(pts_diff[:, 0])
    dty = np.sum(pts_diff[:, 1])

    jacobian = 2 * np.array([dtx, dty, dtheta]) / pts_size
    return jacobian


def getJacobianSE2(params, in_pts, out_pts, pts_size):
    (trans_mat, rot_mat, scale_mat) = getDecomposedSE2Matrices(params)
    rec_pts = util.dehomogenize(trans_mat * rot_mat * scale_mat * in_pts)
    out_pts = np.mat(util.dehomogenize(out_pts))
    in_pts = np.mat(util.dehomogenize(in_pts))
    pts_diff = np.mat(rec_pts - out_pts).transpose()
    theta = params[2]
    s = params[3]
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    ds_mat = np.mat([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])
    dtheta_mat = np.mat([[-sin_theta, -cos_theta],
                         [cos_theta, -sin_theta]])

    ds = np.dot(np.ravel(pts_diff), np.ravel(ds_mat * in_pts, order='F'))
    dtheta = (1 + s) * np.dot(np.ravel(pts_diff), np.ravel(dtheta_mat * in_pts, order='F'))
    dtx = np.sum(pts_diff[:, 0])
    dty = np.sum(pts_diff[:, 1])

    jacobian = 2 * np.array([dtx, dty, dtheta, ds]) / pts_size
    # print 'params:\n', params
    # print 'ds_mat:\n', ds_mat
    # print 'in_pts:\n', in_pts
    # print 'out_pts:\n', out_pts
    # print 'rec_pts:\n', rec_pts
    # print 'pts_diff:\n', pts_diff
    # print 'jacobian:\n', jacobian
    return jacobian


def getJacobianAffine(params, in_pts, out_pts, pts_size):
    (trans_mat, rot_mat, scale_mat, shear_mat) = getDecomposedAffineMatrices(params)
    rec_pts = trans_mat * rot_mat * scale_mat * shear_mat * in_pts
    shear_rec_pts = util.dehomogenize(shear_mat * in_pts)

    out_pts = np.mat(util.dehomogenize(out_pts))
    in_pts = np.mat(util.dehomogenize(in_pts))
    pts_diff = np.mat(rec_pts - out_pts).transpose()

    da = np.dot(pts_diff[0, :], in_pts[0, :])
    db = np.dot(pts_diff[0, :], in_pts[1, :])

    theta = params[2]
    s = params[3]
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    ds_mat = np.mat([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])
    dtheta_mat = np.mat([[-sin_theta, -cos_theta],
                         [cos_theta, -sin_theta]])

    ds = np.dot(np.ravel(pts_diff), np.ravel(ds_mat * shear_rec_pts, order='F'))
    dtheta = (1 + s) * np.dot(np.ravel(pts_diff), np.ravel(dtheta_mat * shear_rec_pts, order='F'))
    dtx = np.sum(pts_diff[:, 0])
    dty = np.sum(pts_diff[:, 1])

    jacobian = 2 * np.array([dtx, dty, dtheta, ds, da, db]) / pts_size
    return jacobian


def getSSDTrans(params, in_pts, out_pts, pts_size):
    trans_mat = getTranslationMatrix(params[0], params[1])
    rec_pts = trans_mat * in_pts
    rec_error = np.sum(np.square(rec_pts - out_pts)) / pts_size
    return rec_error


def getSSDRotate(theta, in_pts, out_pts, pts_size):
    rot_mat = getRotationMatrix(theta)
    rec_pts = rot_mat * in_pts
    rec_error = np.sum(np.square(rec_pts - out_pts)) / pts_size
    return rec_error


def getSSDScale(s, in_pts, out_pts, pts_size):
    scale_mat = getScalingMatrix(s)
    rec_pts = scale_mat * in_pts
    rec_error = np.sum(np.square(rec_pts - out_pts)) / pts_size
    return rec_error


def getSSDShear(params, in_pts, out_pts, pts_size):
    # print 'getSSDShear:: params:', params
    params_len = len(params)
    # print 'getSSDShear:: params_len:', params_len
    if params_len < 2:
        # print'getSSDShear:: Invalid params provided'
        params = params[0]
        # sys.exit()
    shear_mat = getShearingMatrix(params[0], params[1])
    rec_pts = shear_mat * in_pts
    rec_error = np.sum(np.square(rec_pts - out_pts)) / pts_size
    return rec_error


def getSSDProj(params, in_pts, out_pts, pts_size):
    # print 'getSSDShear:: params:', params
    params_len = len(params)
    # print 'getSSDShear:: params_len:', params_len
    if params_len < 2:
        # print'getSSDShear:: Invalid params provided'
        params = params[0]
        # sys.exit()
    proj_mat = getProjectionMatrix(params[0], params[1])
    rec_pts = util.homogenize(util.dehomogenize(proj_mat * in_pts))
    rec_error = np.sum(np.square(rec_pts - out_pts)) / pts_size
    return rec_error


def getSSDRT(params, in_pts, out_pts, pts_size):
    (trans_mat, rot_mat) = getDecomposedRTMatrices(params)
    rec_pts = trans_mat * rot_mat * in_pts
    rec_error = np.sum(np.square(rec_pts - out_pts)) / pts_size
    return rec_error


def getSSDTranscaling(params, in_pts, out_pts, pts_size):
    trs_mat = getTranscalingMatrix(params[0], params[1], params[2])
    rec_pts = trs_mat * in_pts
    return np.sum(np.square(rec_pts - out_pts)) / pts_size


def getSSDSE2(params, in_pts, out_pts, pts_size):
    (trans_mat, rot_mat, scale_mat) = getDecomposedSE2Matrices(params)
    rec_pts = trans_mat * rot_mat * scale_mat * in_pts
    rec_error = np.sum(np.square(rec_pts - out_pts)) / pts_size
    return rec_error


def getSSDAffine(params, in_pts, out_pts, pts_size):
    (trans_mat, rot_mat, scale_mat, shear_mat) = getDecomposedAffineMatrices(params)
    rec_pts = trans_mat * rot_mat * scale_mat * shear_mat * in_pts
    rec_error = np.sum(np.square(rec_pts - out_pts)) / pts_size
    return rec_error


def getSSDRTInverse(params, in_pts, out_pts, pts_size):
    (trans_mat, rot_mat) = getDecomposedRTMatrices(params)
    rec_pts = rot_mat * trans_mat * in_pts
    rec_error = np.sum(np.square(rec_pts - out_pts)) / pts_size
    return rec_error


def getSSDSE2Inverse(params, in_pts, out_pts, pts_size):
    (trans_mat, rot_mat, scale_mat) = getDecomposedSE2Matrices(params)
    rec_pts = scale_mat * rot_mat * trans_mat * in_pts
    rec_error = np.sum(np.square(rec_pts - out_pts)) / pts_size
    return rec_error


def getSSDAffineInverse(params, in_pts, out_pts, pts_size):
    (trans_mat, rot_mat, scale_mat, shear_mat) = getDecomposedAffineMatrices(params)
    rec_pts = shear_mat * scale_mat * rot_mat * trans_mat * in_pts
    rec_error = np.sum(np.square(rec_pts - out_pts)) / pts_size
    return rec_error


def getNormalizedPoints(pts):
    centroid = np.mean(pts, axis=1)
    trans_mat = getTranslationMatrix(-centroid[0], -centroid[1])
    trans_mat_inv = getTranslationMatrix(centroid[0], centroid[1])
    trans_pts = util.dehomogenize(trans_mat * util.homogenize(pts))
    mean_dist = np.mean(np.sqrt(np.sum(np.square(trans_pts), axis=1)))
    if mean_dist == 0:
        print 'Error in getNormalizedPoints:: mean distance between the given points is zero: ', pts
        sys.exit()
    norm_scale = math.sqrt(2) / mean_dist
    scale_mat = getScalingMatrix(norm_scale - 1)
    scale_mat_inv = getScalingMatrix(1.0 / norm_scale - 1)
    norm_pts = util.dehomogenize(scale_mat * trans_mat * util.homogenize(pts))
    norm_mat_inv = trans_mat_inv * scale_mat_inv
    # print 'pts:\n', pts
    # print 'centroid:\n', centroid
    # print 'trans_mat:\n', trans_mat
    # print 'trans_mat_inv:\n', trans_mat_inv
    # print 'trans_pts:\n', trans_pts
    # print 'mean_dist:', mean_dist
    # print 'norm_scale:', norm_scale
    # print 'scale_mat:\n', scale_mat
    # print 'scale_mat_inv:\n', scale_mat_inv
    # print 'norm_mat_inv:\n', norm_mat_inv
    return norm_pts, norm_mat_inv


def getAugmentingSE2Params(init_params, final_params):
    tx = final_params[0] - init_params[0]
    ty = final_params[1] - init_params[1]
    theta = final_params[2] - init_params[2]
    s = ((1 + final_params[3]) / (1 + init_params[3])) - 1
    return [tx, ty, theta, s]


def getDecomposedRTMatrices(params):
    tx = params[0]
    ty = params[1]
    theta = params[2]

    trans_mat = getTranslationMatrix(tx, ty)
    rot_mat = getRotationMatrix(theta)

    return (trans_mat, rot_mat)


def getDecomposedSE2Matrices(params):
    tx = params[0]
    ty = params[1]
    theta = params[2]
    s = params[3]

    trans_mat = getTranslationMatrix(tx, ty)
    rot_mat = getRotationMatrix(theta)
    scale_mat = getScalingMatrix(s)

    return (trans_mat, rot_mat, scale_mat)


def getDecomposedAffineMatrices(params):
    tx = params[0]
    ty = params[1]
    theta = params[2]
    s = params[3]
    a = params[4]
    b = params[5]

    trans_mat = getTranslationMatrix(tx, ty)
    rot_mat = getRotationMatrix(theta)
    scale_mat = getScalingMatrix(s)
    shear_mat = getShearingMatrix(a, b)

    return (trans_mat, rot_mat, scale_mat, shear_mat)


def getJaccardError(img1, img2):
    img_intersection = np.sum(np.bitwise_and(img1, img2))
    img_union = np.sum(np.bitwise_or(img1, img2))
    jacc_error = float(img_intersection) / float(img_union)
    return jacc_error


def getJaccardErrorPts(pts1, pts2, img_size):
    img1 = getBinaryPtsImage2(img_size, pts1)
    img2 = getBinaryPtsImage2(img_size, pts2)

    img_intersection = np.sum(np.bitwise_and(img1, img2))
    img_union = np.sum(np.bitwise_or(img1, img2))
    jacc_error = 1.0 - float(img_intersection) / float(img_union)
    return jacc_error


def getBinaryPtsImage(img_shape, pts):
    img_shape = img_shape[0:2]
    # idx_ceil = np.ceil(pts[0, :]).astype(np.int16)
    # idy_ceil = np.ceil(pts[1, :]).astype(np.int16)
    idx_floor = np.floor(pts[0, :]).astype(np.int16)
    idy_floor = np.floor(pts[1, :]).astype(np.int16)
    idx_ceil = idx_floor + 1
    idy_ceil = idy_floor + 1

    # print 'idx:\n', idx
    # print 'idy:\n', idy
    bin_img = np.zeros(img_shape, dtype=np.uint8)
    bin_img[idy_ceil, idx_ceil] = 255
    bin_img[idy_ceil, idx_floor] = 255
    bin_img[idy_floor, idx_floor] = 255
    bin_img[idy_floor, idx_ceil] = 255

    return bin_img


def getNormalizedUnitSquarePts(resx=100, resy=100, c=1.0):
    pts_arr = np.mat(np.zeros((2, resy * resx)))
    pt_id = 0
    for x in np.linspace(-c, c, resx):
        for y in np.linspace(-c, c, resy):
            pts_arr[0, pt_id] = x
            pts_arr[1, pt_id] = y
            pt_id += 1
    corners = np.mat([[-c, c, c, -c], [-c, -c, c, c]])
    return pts_arr, corners


# def readTrackingData(filename):
# if not os.path.isfile(filename):
# print "Tracking data file not found:\n ", filename
# sys.exit()
#
# data_file = open(filename, 'r')
#     data_file.readline()
#     lines = data_file.readlines()
#     no_of_lines = len(lines)
#     data_array = np.empty([no_of_lines, 8])
#     line_id = 0
#     for line in lines:
#         words = line.split()
#         if len(words) != 9:
#             msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
#             raise SyntaxError(msg)
#         words = words[1:]
#         coordinates = []
#         for word in words:
#             coordinates.append(float(word))
#         data_array[line_id, :] = coordinates
#         line_id += 1
#     data_file.close()
#     return data_array


def shiftProjectiveCenterToOrigin(corners):
    x1 = corners[0, 0]
    y1 = corners[1, 0]
    x2 = corners[0, 1]
    y2 = corners[1, 1]
    x3 = corners[0, 2]
    y3 = corners[1, 2]
    x4 = corners[0, 3]
    y4 = corners[1, 3]

    m1 = (y1 - y3) / (x1 - x3)
    b1 = y1 - m1 * x1
    m2 = (y2 - y4) / (x2 - x4)
    b2 = y2 - m2 * x2

    proj_center_x = -(b1 - b2) / (m1 - m2)
    proj_center_y = m1 * proj_center_x + b1

    trans_mat = np.mat(
        [[1, 0, -proj_center_x],
         [0, 1, -proj_center_y],
         [0, 0, 1]]
    )
    trans_mat_inv = np.mat(
        [[1, 0, proj_center_x],
         [0, 1, proj_center_y],
         [0, 0, 1]]
    )
    shifted_corners = util.dehomogenize(trans_mat * util.homogenize(corners))
    return (shifted_corners, trans_mat_inv)


def shiftGeometricCenterToOrigin(corners):
    geom_center_x = np.mean(corners[0, :])
    geom_center_y = np.mean(corners[1, :])

    trans_mat = np.mat(
        [[1, 0, -geom_center_x],
         [0, 1, -geom_center_y],
         [0, 0, 1]]
    )
    trans_mat_inv = np.mat(
        [[1, 0, geom_center_x],
         [0, 1, geom_center_y],
         [0, 0, 1]]
    )
    shifted_corners = util.dehomogenize(trans_mat * util.homogenize(corners))
    return (shifted_corners, trans_mat_inv)


def getRectangularApproximation(corners_in, img_size=None):
    center_x = np.mean(corners_in[0, :])
    center_y = np.mean(corners_in[1, :])

    mean_width = (abs(corners_in[0, 0] - center_x) + abs(corners_in[0, 1] - center_x)
                  + abs(corners_in[0, 2] - center_x) + abs(corners_in[0, 3] - center_x)) / 2.0
    mean_height = (abs(corners_in[1, 0] - center_y) + abs(corners_in[1, 1] - center_y)
                   + abs(corners_in[1, 2] - center_y) + abs(corners_in[1, 3] - center_y)) / 2.0

    # mean_height2 = (abs(corners_in[1, 3] - corners_in[1, 0])
    #                 + abs(corners_in[1, 2] - corners_in[1, 1])) / 2.0
    # mean_width2 = (abs(corners_in[0, 1] - corners_in[0, 0])
    #                + abs(corners_in[0, 2] - corners_in[0, 3])) / 2.0
    # print 'mean_height: ', mean_height
    # print 'mean_width: ', mean_width
    # print 'base_x: ', base_x
    # print 'base_y: ', base_y
    min_x = center_x - mean_width / 2.0
    max_x = center_x + mean_width / 2.0
    min_y = center_y - mean_height / 2.0
    max_y = center_y + mean_height / 2.0

    # min_x2 = center_x - mean_width2 / 2.0
    # max_x2 = center_x + mean_width2 / 2.0
    # min_y2 = center_y - mean_height2 / 2.0
    # max_y2 = center_y + mean_height2 / 2.0

    rect_corners = np.asarray([[min_x, max_x, max_x, min_x],
                               [min_y, min_y, max_y, max_y]])
    # rect_corners2 = np.asarray([[min_x2, max_x2, max_x2, min_x2],
    #                             [min_y2, min_y2, max_y2, max_y2]])
    # max_rec_error=img_size[0]**2 + img_size[1]**2
    # rec_error = np.sum(np.square(corners_in - rect_corners)) / (4.0)
    # rec_error2 = np.sum(np.square(corners_in - rect_corners2)) / (4.0)
    # rec_ratio = rec_error / rec_error2
    # print 'rec_error: ', rec_error
    # print 'rec_error2: ', rec_error2
    # print '\trec_ratio: ', rec_ratio
    # if img_size is not None:
    #     jac_error = 1 - getJaccardErrorPts(corners_in, rect_corners, img_size)
    #     jac_error2 = 1 - getJaccardErrorPts(corners_in, rect_corners2, img_size)
    #     print 'jac_error: ', jac_error
    #     print 'jac_error2: ', jac_error2
    #     if jac_error2 !=0:
    #         jac_ratio = jac_error / jac_error2
    #         print '\tjac_ratio: ', jac_ratio
    return rect_corners


def computeHomographyLS(in_pts, out_pts):
    num_pts = in_pts.shape[1]
    A = np.zeros((num_pts * 2, 8), dtype=np.float64)
    b = np.zeros((num_pts * 2, 1), dtype=np.float64)
    for j in xrange(num_pts):
        x1 = in_pts[0, j]
        y1 = in_pts[1, j]
        x2 = out_pts[0, j]
        y2 = out_pts[1, j]

        r1 = 2 * j
        b[r1] = -y2
        A[r1, 3] = -x1
        A[r1, 4] = -y1
        A[r1, 5] = -1
        A[r1, 6] = x1 * y2
        A[r1, 7] = y1 * y2

        r2 = 2 * j + 1
        b[r2] = x2
        A[r2, 0] = x1
        A[r2, 1] = y1
        A[r2, 2] = 1
        A[r2, 6] = -x1 * x2
        A[r2, 7] = -y1 * x2

    U, S, V = np.linalg.svd(A, full_matrices=False)
    b2 = np.mat(U.transpose()) * np.mat(b)
    y = np.zeros(b2.shape, dtype=np.float64)
    for j in xrange(b2.shape[0]):
        y[j] = b2[j] / S[j]
    x = np.mat(V.transpose()) * np.mat(y)
    hom_mat = np.zeros((3, 3), dtype=np.float64)
    hom_mat[0, 0] = x[0, 0]
    hom_mat[0, 1] = x[1, 0]
    hom_mat[0, 2] = x[2, 0]
    hom_mat[1, 0] = x[3, 0]
    hom_mat[1, 1] = x[4, 0]
    hom_mat[1, 2] = x[5, 0]
    hom_mat[2, 0] = x[6, 0]
    hom_mat[2, 1] = x[7, 0]
    hom_mat[2, 2] = 1.0

    # print 'x:\n', x
    # print 'ls_hom_mat:\n', hom_mat

    return np.mat(hom_mat)


def computeAffineLS(in_pts, out_pts):
    num_pts = in_pts.shape[1]
    A = np.zeros((num_pts * 2, 6), dtype=np.float64)
    b = np.zeros((num_pts * 2, 1), dtype=np.float64)
    for j in xrange(num_pts):
        r1 = 2 * j
        b[r1] = out_pts[0, j]
        A[r1, 0] = in_pts[0, j]
        A[r1, 1] = in_pts[1, j]
        A[r1, 2] = 1

        r2 = 2 * j + 1
        b[r2] = out_pts[1, j]
        A[r2, 3] = in_pts[0, j]
        A[r2, 4] = in_pts[1, j]
        A[r2, 5] = 1
    U, S, V = np.linalg.svd(A, full_matrices=False)

    # print 'U.shape: ', U.shape
    # print 'S.shape: ', S.shape
    # print 'V.shape: ', V.shape
    # rec_A = np.mat(U) * np.mat(np.diag(S)) * np.mat(V)    #
    # rec_err_mat = rec_A - A

    b2 = np.mat(U.transpose()) * np.mat(b)
    y = np.zeros(b2.shape, dtype=np.float64)
    for j in xrange(b2.shape[0]):
        y[j] = b2[j] / S[j]
    x = np.mat(V.transpose()) * np.mat(y)

    affine_b_rec = A * x
    affine_b_rec_error = math.sqrt(np.sum(np.square(affine_b_rec - b)) / (num_pts * 4))
    # print 'affine U:\n', U
    # print 'affine S:\n', S
    # print 'affine V:\n', V
    #
    # print 'affine_b:\n', b
    # print 'affine_b2:\n', b2
    # print 'affine_b_rec:\n', affine_b_rec
    # print 'affine_b_rec_error: ', affine_b_rec_error

    # printMatrixToFile(in_pts, 'in_pts', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    # printMatrixToFile(out_pts, 'out_pts', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')    #
    # printMatrixToFile(A, 'A', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    # printMatrixToFile(U, 'U', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    # printMatrixToFile(V, 'V', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    # printVectorFile(S, 'S', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    # printMatrixToFile(rec_A, 'rec_A', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    # printMatrixToFile(rec_err_mat, 'rec_err_mat', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    # printVectorFile(b, 'b', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    # printMatrixToFile(b2, 'b2', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    # printVectorFile(y, 'y', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')
    # printMatrixToFile(x, 'x', './C++/log/py_log.txt', fmt='{:15.9f}', mode='a', sep='\t')

    x_mat = x.reshape(2, 3)
    affine_mat = np.zeros((3, 3), dtype=np.float64)

    affine_mat[0, :] = x_mat[0, :]
    affine_mat[1, :] = x_mat[1, :]
    affine_mat[2, 2] = 1.0
    return np.mat(affine_mat)


def computeSimilarityLS(in_pts, out_pts):
    num_pts = in_pts.shape[1]
    A = np.zeros((num_pts * 2, 4), dtype=np.float64)
    b = np.zeros((num_pts * 2, 1), dtype=np.float64)
    for j in xrange(num_pts):
        r1 = 2 * j
        b[r1] = out_pts[0, j] - in_pts[0, j]
        A[r1, 0] = 1
        A[r1, 2] = in_pts[0, j]
        A[r1, 3] = -in_pts[1, j]

        r2 = 2 * j + 1
        b[r2] = out_pts[1, j] - in_pts[1, j]
        A[r2, 1] = 1
        A[r2, 2] = in_pts[1, j]
        A[r2, 3] = in_pts[0, j]

    U, S, V = np.linalg.svd(A, full_matrices=False)
    #
    # print 'U.shape: ', U.shape
    # print 'S.shape: ', S.shape
    # print 'V.shape: ', V.shape

    b2 = np.mat(U.transpose()) * np.mat(b)
    y = np.zeros(b2.shape, dtype=np.float64)
    for j in xrange(b2.shape[0]):
        y[j] = b2[j] / S[j]
    x = np.mat(V.transpose()) * np.mat(y)

    sim_mat = np.zeros((3, 3), dtype=np.float64)
    sim_mat[0, 0] = 1 + x[2]
    sim_mat[0, 1] = -x[3]
    sim_mat[0, 2] = x[0]
    sim_mat[1, 0] = x[3]
    sim_mat[1, 1] = 1 + x[2]
    sim_mat[1, 2] = x[1]
    sim_mat[2, 2] = 1.0
    return np.mat(sim_mat)


def computeTranscalingyLS(in_pts, out_pts):
    num_pts = in_pts.shape[1]
    A = np.zeros((num_pts * 2, 3), dtype=np.float64)
    b = np.zeros((num_pts * 2, 1), dtype=np.float64)
    for j in xrange(num_pts):
        r1 = 2 * j
        b[r1] = out_pts[0, j] - in_pts[0, j]
        A[r1, 0] = 1
        A[r1, 2] = in_pts[0, j]

        r2 = 2 * j + 1
        b[r2] = out_pts[1, j] - in_pts[1, j]
        A[r2, 1] = 1
        A[r2, 2] = in_pts[1, j]

    U, S, V = np.linalg.svd(A, full_matrices=False)

    # print 'U.shape: ', U.shape
    # print 'S.shape: ', S.shape
    # print 'V.shape: ', V.shape

    b2 = np.mat(U.transpose()) * np.mat(b)
    y = np.zeros(b2.shape, dtype=np.float64)
    for j in xrange(b2.shape[0]):
        y[j] = b2[j] / S[j]
    x = np.mat(V.transpose()) * np.mat(y)

    trs_mat = np.zeros((3, 3), dtype=np.float64)
    trs_mat[0, 0] = 1 + x[2]
    trs_mat[0, 2] = x[0]
    trs_mat[1, 1] = 1 + x[2]
    trs_mat[1, 2] = x[1]
    trs_mat[2, 2] = 1.0
    return np.mat(trs_mat)


def getNCCPoints2(vals1, vals2):
    num = np.sum(np.multiply(vals1, vals2))
    den = np.sqrt(np.sum(np.square(vals1)) * np.sum(np.square(vals2)))
    return num / den


def getNCCPoints(vals1, vals2):
    vals1 = vals1 - np.mean(vals1)
    vals2 = vals2 - np.mean(vals2)
    num = np.sum(np.multiply(vals1, vals2))
    den = np.sqrt(np.sum(np.square(vals1)) * np.sum(np.square(vals2)))
    return num / den


def getSSIMPoints(vals1, vals2):
    return np.sum(np.divide(np.multiply(vals1, vals2), (np.square(vals1) + np.square(vals2))))


def getSSDPoints(vals1, vals2):
    return np.sum(np.square(vals1 - vals2))


def getMIPoints(img1, img2, norm_max=7, no_of_bins=0):
    # print 'getMIPoints'
    h12, h1, h2 = getJointHistogram(img1, img2, norm_max, no_of_bins)
    mi = 0
    for r in xrange(norm_max):
        for t in xrange(norm_max):
            if h12[r, t] == 0 or h12[r, t] == 0 or h2[t] == 0:
                continue
            mi += h12[r, t] * math.log(h12[r, t] / (h1[r] * h2[t]))
    # print 'h12: ', h12
    # print 'h1: ', h1
    # print 'h2: ', h2
    # print 'mi: ', mi
    # k=raw_input('press any key')
    return mi


def getJointHistogram(x, y, norm_max=7, no_of_bins=0):
    # print 'getJointHistogram'
    if no_of_bins <= 0:
        no_of_bins = int(norm_max + 1)

    # print 'no_of_bins: ', no_of_bins
    # print 'norm_max: ', norm_max

    x = np.array(x)
    y = np.array(y)

    x_norm = x * (norm_max / 255.0)
    y_norm = y * (norm_max / 255.0)
    x_norm_rnd = np.rint(x_norm)
    y_norm_rnd = np.rint(y_norm)

    # np.savetxt('x.txt', x, fmt='%10.5f')
    # np.savetxt('y.txt', y, fmt='%10.5f')
    # np.savetxt('x_norm.txt', x_norm, fmt='%10.5f')
    # np.savetxt('y_norm.txt', y_norm, fmt='%10.5f')
    # np.savetxt('x_norm_rnd.txt', x_norm_rnd, fmt='%10.5f')
    # np.savetxt('y_norm_rnd.txt', y_norm_rnd, fmt='%10.5f')


    # print 'x: ', x
    # print 'y: ', y
    # print 'x_norm: ', x_norm
    # print 'y_norm: ', y_norm
    # print 'x_norm_rnd: ', x_norm_rnd
    # print 'y_norm_rnd: ', y_norm_rnd

    # k=raw_input('naziora')


    npts = x.shape[0]
    # k=raw_input('npts')
    hxy = np.zeros((no_of_bins, no_of_bins))
    # k=raw_input('hxy')
    hx = np.zeros((no_of_bins, 1))
    # k=raw_input('hx')
    hy = np.zeros((no_of_bins, 1))
    # k=raw_input('hy')


    # print 'npts: ', npts


    # print 'Starting loop'
    for pt_id in xrange(npts):
        # print 'pt_id: ', pt_id
        # print 'x_val: ', x_norm_rnd[pt_id]
        # print 'y_val: ', y_norm_rnd[pt_id]
        hxy[x_norm_rnd[pt_id], y_norm_rnd[pt_id]] += 1
        hx[x_norm_rnd[pt_id]] += 1
        hy[y_norm_rnd[pt_id]] += 1

    # print 'Done'
    # print 'hxy: ', hxy
    # print 'hx: ', hx
    # print 'hy: ', hy
    hxy /= npts
    hx /= npts
    hy /= npts
    return hxy, hx, hy


def applyFilter(src_img, filter_type, kernel_size=5, scale_factor=1,
                gabor_params=(1.0, 0.0, 11.0, 1.0), canny_params=(20, 4), gauss_std=3,
                dog_params=(3.0, 3.8, 2.5)):
    if filter_type == 'gauss':
        print 'Applying gaussian blur...'
        filtered_img = cv2.GaussianBlur(src_img, (kernel_size, kernel_size), gauss_std)
    elif filter_type == 'norm_box':
        filtered_img = cv2.blur(src_img, (kernel_size, kernel_size))
    elif filter_type == 'bilateral':
        filtered_img = cv2.bilateralFilter(src_img, kernel_size, 100, 100)
    elif filter_type == 'median':
        filtered_img = cv2.medianBlur(src_img, kernel_size)
    elif filter_type == 'box':
        filtered_img = cv2.boxFilter(src_img, -1, (kernel_size, kernel_size))
    elif filter_type == 'gabor':
        print 'Applying gabor filter...'
        gabor_kernel = cv2.getGaborKernel(ksize=(kernel_size, kernel_size),
                                          sigma=gabor_params[0], theta=gabor_params[1],
                                          lambd=gabor_params[2], gamma=gabor_params[3])
        filtered_img = cv2.filter2D(src_img.astype(np.float64), -1, gabor_kernel)
    elif filter_type == 'laplacian':
        filtered_img = cv2.Laplacian(src_img.astype(np.float64), -1, ksize=kernel_size, scale=scale_factor)
    elif filter_type == 'sobel':
        filtered_img = cv2.Sobel(src_img.astype(np.float64), -1, ksize=kernel_size, scale=scale_factor,
                                 dx=1, dy=0)
    elif filter_type == 'scharr':
        filtered_img = cv2.Scharr(src_img.astype(np.float64), -1, scale=scale_factor,
                                  dx=1, dy=0)
    elif filter_type == 'canny':
        filtered_img = cv2.Canny(src_img.astype(np.uint8), threshold1=canny_params[0],
                                 threshold2=canny_params[0] * canny_params[1])
    elif filter_type == 'LoG':
        gauss_img = cv2.GaussianBlur(src_img.astype(np.float64), ksize=(kernel_size, kernel_size), sigmaX=gauss_std)
        filtered_img = cv2.Laplacian(gauss_img, -1, ksize=kernel_size, scale=scale_factor)
    elif filter_type == 'DoG':
        src_img = src_img.astype(np.uint8)
        ex_img = cv2.GaussianBlur(src_img, ksize=(kernel_size, kernel_size),
                                  sigmaX=dog_params[0])
        in_img = cv2.GaussianBlur(src_img, (kernel_size, kernel_size),
                                  sigmaX=dog_params[1])
        filtered_img = ex_img - dog_params[2] * in_img
    else:
        filtered_img = src_img

    return filtered_img.astype(np.float64)
