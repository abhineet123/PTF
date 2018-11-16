from DecompUtils import *


def reformatHomFile(in_fname, out_fname, decomp_type='inverse'):
    if not os.path.isfile(in_fname):
        print "Homography data file not found:\n ", in_fname
        sys.exit()

    in_file = open(in_fname, 'r')
    out_file = open(out_fname, 'w')

    lines = in_file.readlines()
    line_id = 0
    frame_id = 0
    hom_params_list=[]
    hom_mat = np.zeros((3, 3), dtype=np.float64)
    for line in lines:
        words = line.split()
        if len(words) == 1:
            frame_id += 1
            continue
        elif len(words) != 10:
            raise StandardError('Unexpected formatting on line {:d}:\n{:s}'.format(line_id, line))

        words = words[1:]
        mat_entries = []
        for word in words:
            mat_entries.append(float(word))

        hom_mat[0, 0] = mat_entries[0]
        hom_mat[0, 1] = mat_entries[1]
        hom_mat[0, 2] = mat_entries[2]
        hom_mat[1, 0] = mat_entries[3]
        hom_mat[1, 1] = mat_entries[4]
        hom_mat[1, 2] = mat_entries[5]
        hom_mat[2, 0] = mat_entries[6]
        hom_mat[2, 1] = mat_entries[7]
        hom_mat[2, 2] = mat_entries[8]

        if decomp_type == 'inverse':
            hom_params = getHomographyParamsInverse(hom_mat)
        else:
            hom_params = getHomographyParamsForward(hom_mat)

        hom_params_list.append(hom_params)
        out_file.write('{:d}'.format(frame_id))
        for param in hom_params:
            out_file.write('\t{:12.8f}'.format(param))
        out_file.write('\n')

        line_id += 1
    in_file.close()
    out_file.close()
    return np.array(hom_params_list)


def getHomographyParamsInverse(hom_mat):
    affine_mat, proj_mat, hom_rec_mat = decomposeHomographyInverse(hom_mat)
    trans_mat, rt_mat, se2_mat, affine_rec_mat, affine_params = decomposeAffineInverse(affine_mat)
    tx = affine_params[0]
    ty = affine_params[1]
    theta = affine_params[2]
    scale = affine_params[3]
    a = affine_params[4]
    b = affine_params[5]
    v1 = proj_mat[2, 0]
    v2 = proj_mat[2, 1]

    return tx, ty, theta, scale, a, b, v1, v2


def getHomographyParamsForward(hom_mat):
    affine_mat, proj_mat, hom_rec_mat = decomposeHomographyForward(hom_mat)
    trans_mat, rt_mat, se2_mat, affine_rec_mat, affine_params = decomposeAffineForward(affine_mat)
    tx = affine_params[0]
    ty = affine_params[1]
    theta = affine_params[2]
    scale = affine_params[3]
    a = affine_params[4]
    b = affine_params[5]
    v1 = proj_mat[2, 0]
    v2 = proj_mat[2, 1]

    return tx, ty, theta, scale, a, b, v1, v2


def decomposeAffineInverse(affine_mat):
    a1 = affine_mat[0, 0]
    a2 = affine_mat[0, 1]
    a3 = affine_mat[0, 2]
    a4 = affine_mat[1, 0]
    a5 = affine_mat[1, 1]
    a6 = affine_mat[1, 2]

    a = (a1 * a5 - a2 * a4) / (a5 * a5 + a4 * a4) - 1
    b = (a2 * a5 + a1 * a4) / (a5 * a5 + a4 * a4)
    a7 = (a3 - a6 * b) / (1 + a)
    s = np.sqrt(a5 * a5 + a4 * a4) - 1
    tx = (a4 * a6 + a5 * a7) / (a5 * a5 + a4 * a4)
    ty = (a5 * a6 - a4 * a7) / (a5 * a5 + a4 * a4)
    cos_theta = a5 / (1 + s)
    sin_theta = a4 / (1 + s)

    if cos_theta < 0 and sin_theta < 0:
        cos_theta = -cos_theta
        sin_theta = -sin_theta
        s = -(s + 2)

    theta1 = math.acos(cos_theta)
    # theta2 = math.asin(sin_theta)
    #
    # theta_sum = theta1 + theta2
    # theta_diff = theta1 - theta2
    #
    # print 'cos_theta: ', cos_theta
    # print 'sin_theta: ', sin_theta
    #
    # print 'theta1: ', theta1
    # print 'theta2: ', theta2
    # print 'theta_sum: ', theta_sum
    # print 'theta_diff: ', theta_diff
    # print 's: ', s

    params = [tx, ty, theta1, s, a, b]

    # print 'inverse: cos_theta: ', cos_theta, '\t theta: ', math.acos(cos_theta)
    # print 'inverse: sin_theta: ', sin_theta, '\t theta: ', math.asin(sin_theta)
    # print 'inverse: s: ', s
    # print 'inverse: a: ', a
    # print 'inverse: b: ', b

    (trans_mat, rot_mat, scale_mat, shear_mat) = getDecomposedAffineMatrices(params)

    # rot_mat = np.mat(
    # [[cos_theta, - sin_theta, 0],
    # [sin_theta, cos_theta, 0],
    # [0, 0, 1]]
    # )

    # print 'decomposeAffineInverse: shear_mat:\n', shear_mat

    rt_mat = rot_mat * trans_mat
    se2_mat = scale_mat * rt_mat
    affine_rec_mat = shear_mat * se2_mat

    return trans_mat, rt_mat, se2_mat, affine_rec_mat, params


def decomposeAffineForward(affine_mat):
    a1 = affine_mat[0, 0]
    a2 = affine_mat[0, 1]
    a3 = affine_mat[0, 2]
    a4 = affine_mat[1, 0]
    a5 = affine_mat[1, 1]
    a6 = affine_mat[1, 2]

    b = (a1 * a2 + a4 * a5) / (a1 * a5 - a2 * a4)
    a = (b * a1 - a4) / a2 - 1
    s = np.sqrt(a1 * a1 + a4 * a4) / (1 + a) - 1
    tx = a3
    ty = a6
    cos_theta = a1 / ((1 + s) * (1 + a))
    sin_theta = a4 / ((1 + s) * (1 + a))

    # print 'cos_theta: ', cos_theta
    # print 'sin_theta: ', sin_theta

    if cos_theta > 1.0:
        cos_theta = 0.0

    theta = math.acos(cos_theta)
    params = [tx, ty, theta, s, a, b]

    # print 'forward: cos_theta: ', cos_theta, '\t theta: ', math.acos(cos_theta)
    # print 'forward: sin_theta: ', sin_theta, '\t theta: ', math.asin(sin_theta)
    # print 'forward: s: ', s
    # print 'forward: a: ', a
    # print 'forward: b: ', b

    (trans_mat, rot_mat, scale_mat, shear_mat) = getDecomposedAffineMatrices(params)

    rt_mat = trans_mat * rot_mat
    se2_mat = rt_mat * scale_mat
    affine_rec_mat = se2_mat * shear_mat

    return trans_mat, rt_mat, se2_mat, affine_rec_mat, params


def decomposeHomographyForward(hom_mat):
    h1 = hom_mat[0, 0]
    h2 = hom_mat[0, 1]
    h3 = hom_mat[0, 2]
    h4 = hom_mat[1, 0]
    h5 = hom_mat[1, 1]
    h6 = hom_mat[1, 2]
    h7 = hom_mat[2, 0]
    h8 = hom_mat[2, 1]

    a1 = h1 - h3 * h7
    a2 = h2 - h3 * h8
    a3 = h3
    a4 = h4 - h6 * h7
    a5 = h5 - h6 * h8
    a6 = h6
    affine_mat = np.mat(
        [[a1, a2, a3],
         [a4, a5, a6],
         [0, 0, 1]],
        dtype=np.float64)

    v1 = h7
    v2 = h8
    proj_mat = np.mat(
        [[1, 0, 0],
         [0, 1, 0],
         [v1, v2, 1]]
    )
    hom_rec_mat = affine_mat * proj_mat
    return affine_mat, proj_mat, hom_rec_mat


def decomposeHomographyInverse(hom_mat):
    h1 = hom_mat[0, 0]
    h2 = hom_mat[0, 1]
    h3 = hom_mat[0, 2]
    h4 = hom_mat[1, 0]
    h5 = hom_mat[1, 1]
    h6 = hom_mat[1, 2]
    h7 = hom_mat[2, 0]
    h8 = hom_mat[2, 1]

    affine_mat = np.mat(
        [[h1, h2, h3],
         [h4, h5, h6],
         [0.0, 0.0, 1.0]],
        dtype=np.float64)

    v1 = (h5 * h7 - h4 * h8 ) / (h1 * h5 - h2 * h4)
    v2 = (h1 * h8 - h2 * h7) / (h1 * h5 - h2 * h4)
    u = 1 - v1 * h3 - v2 * h6
    proj_mat = np.mat(
        [[1, 0, 0],
         [0, 1, 0],
         [v1, v2, u]]
    )
    hom_rec_mat = proj_mat * affine_mat
    return affine_mat, proj_mat, hom_rec_mat


def decomposeHomographyMean(hom_mat):
    (affine_mat_inv, proj_mat_inv, hom_rec_mat_inv) = decomposeHomographyInverse(hom_mat)
    (affine_mat_fwd, proj_mat_fwd, hom_rec_mat_fed) = decomposeHomographyForward(hom_mat)
    affine_mat = (affine_mat_inv + affine_mat_fwd) / 2.0
    proj_mat = (proj_mat_inv + proj_mat_fwd) / 2.0
    hom_rec_mat = (hom_rec_mat_inv + hom_rec_mat_fed) / 2.0
    return affine_mat, proj_mat, hom_rec_mat