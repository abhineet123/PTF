from DecompUtils import *
import scipy.optimize as opt
import time

debug_mode = 0
debug_mode_sh = 0
debug_mode_scale = 0
debug_mode_rot = 0


def computeRotationLS(in_pts, out_pts):
    num_pts = in_pts.shape[1]
    A = np.zeros((num_pts * 4, 2), dtype=np.float64)
    b = np.zeros((num_pts * 4, 1), dtype=np.float64)
    for j in xrange(num_pts):
        x1 = in_pts[0, j]
        y1 = in_pts[1, j]
        x2 = out_pts[0, j]
        y2 = out_pts[1, j]

        r1 = 4 * j
        b[r1] = x2
        A[r1, 0] = x1
        A[r1, 1] = -y1

        r2 = 4 * j + 1
        b[r2] = y2
        A[r2, 0] = y1
        A[r2, 1] = x1

        r3 = 4 * j + 2
        b[r3] = x1
        A[r3, 0] = x2
        A[r3, 1] = y2

        r4 = 4 * j + 3
        b[r4] = y1
        A[r4, 0] = y2
        A[r4, 1] = -x2

    U, S, V = np.linalg.svd(A, full_matrices=False)
    b2 = np.mat(U.transpose()) * np.mat(b)
    y = np.zeros(b2.shape, dtype=np.float64)
    for j in xrange(b2.shape[0]):
        y[j] = b2[j] / S[j]
    x = np.mat(V.transpose()) * np.mat(y)

    # b_rec = A * x
    # b_rec_error = math.sqrt(np.sum(np.square(b_rec - b)) / (num_pts * 4))

    # print 'rot U:\n', U
    # print 'rot S:\n', S
    # print 'rot V:\n', V
    #
    # print 'rot b:\n', b
    # print 'rot b_rec:\n', b_rec
    # print 'rot b_rec_error: ', b_rec_error
    cos_theta = np.fabs(x[0, 0])
    sin_theta = np.fabs(x[1, 0])
    # print 'x.shape: ', x.shape
    # print 'cos_theta: ', cos_theta
    # print 'sin_theta: ', sin_theta
    rot_mat = np.mat(
        [[cos_theta, - sin_theta, 0],
         [sin_theta, cos_theta, 0],
         [0, 0, 1]]
    )
    return np.mat(rot_mat)


def computeRotationBrute(in_pts, out_pts):
    max_theta = np.pi / 2.0
    min_theta = -max_theta
    grid_size = 10000

    # in_pts = trans_mat * np.mat(in_pts)
    in_pts = np.mat(in_pts)
    out_pts = np.mat(out_pts)
    pts_size = np.prod(in_pts.shape)

    # mean_x = np.mean(in_pts[0, :])
    # mean_y = np.mean(in_pts[1, :])
    (opt_theta, opt_ssd, grid, jout) = opt.brute(getSSDRotate, ((min_theta, max_theta),),
                                                 args=(in_pts, out_pts, pts_size), Ns=grid_size,
                                                 full_output=1, disp=True)
    if debug_mode:
        print 'opt_theta: ', opt_theta
        print 'opt_ssd: ', opt_ssd
    opt_rot_mat = getRotationMatrix(opt_theta)
    return opt_rot_mat


def computeScaleBrute(in_pts, out_pts):
    max_scale = 50
    min_scale = 0
    grid_size = 10000

    # in_pts = trans_mat * np.mat(in_pts)
    in_pts = np.mat(in_pts)
    out_pts = np.mat(out_pts)
    pts_size = np.prod(in_pts.shape)

    # mean_x = np.mean(in_pts[0, :])
    # mean_y = np.mean(in_pts[1, :])

    (opt_scale, opt_ssd, grid, jout) = opt.brute(getSSDScale, ((min_scale, max_scale),),
                                                 args=(in_pts, out_pts, pts_size), Ns=grid_size,
                                                 full_output=1, disp=True)
    if debug_mode:
        print 'opt_scale: ', opt_scale
        print 'opt_ssd: ', opt_ssd
    opt_scale_mat = getScalingMatrix(opt_scale)
    return opt_scale_mat


def computeTransOpt(in_pts, out_pts, opt_method, jacobian_func=None, tx0=0, ty0=0):
    min_trans = -500.0
    max_trans = 500.0
    init_guess = np.asarray([tx0, ty0])

    no_of_pts = in_pts.shape[1]
    opt_bounds = ((min_trans, max_trans), (min_trans, max_trans))

    res = opt.minimize(getSSDTrans, init_guess, args=(in_pts, out_pts, no_of_pts),
                       method=opt_method, bounds=opt_bounds, jac=jacobian_func)
    if debug_mode:
        print 'Trans Optimization done: '
        print 'x: ', res.x
        print 'success: ', res.success
        print 'message: ', res.message

    trans_mat = getTranslationMatrix(res.x[0], res.x[1])
    return trans_mat


def computeRotationOpt(in_pts, out_pts, opt_method, jacobian_func=None, theta0=0):
    min_theta0 = -math.pi / 2.0
    max_theta0 = math.pi / 2.0

    opt_bounds = ((min_theta0, max_theta0),)
    init_guess = np.asarray([theta0])

    in_pts = np.mat(in_pts)
    out_pts = np.mat(out_pts)
    no_of_pts = in_pts.shape[1]

    res = opt.minimize(getSSDRotate, init_guess, args=(in_pts, out_pts, no_of_pts),
                       method=opt_method, bounds=opt_bounds, jac=jacobian_func)
    if debug_mode_rot:
        print 'Rotation Optimization done: '
        print 'x: ', res.x
        print 'success: ', res.success
        print 'message: ', res.message

    if res.x.shape:
        opt_theta = res.x[0]
    else:
        opt_theta = res.x
    opt_rot_mat = getRotationMatrix(opt_theta)

    return opt_rot_mat


def computeScaleOpt(in_pts, out_pts, opt_method, jacobian_func=None, scale0=0):
    max_scale = 2.0
    min_scale = 0.0

    opt_bounds = ((min_scale, max_scale),)
    init_guess = np.asarray([scale0])

    in_pts = np.mat(in_pts)
    out_pts = np.mat(out_pts)
    no_of_pts = in_pts.shape[1]

    res = opt.minimize(getSSDScale, init_guess, args=(in_pts, out_pts, no_of_pts),
                       method=opt_method, bounds=opt_bounds, jac=jacobian_func)
    if debug_mode_scale:
        print 'Scale Optimization done: '
        print 'x: ', res.x
        print 'success: ', res.success
        print 'message: ', res.message

    if res.x.shape:
        opt_scale = res.x[0]
    else:
        opt_scale = res.x

    opt_scale_mat = getScalingMatrix(opt_scale)
    return opt_scale_mat


def computeSheartOpt(in_pts, out_pts, opt_method, jacobian_func=None, a0=0, b0=0):
    max_a = 1.0
    min_a = -1.0
    max_b = 1.0
    min_b = -1.0

    opt_bounds = ((min_a, max_a), (min_b, max_b))
    init_guess = np.asarray([a0, b0])

    in_pts = np.mat(in_pts)
    out_pts = np.mat(out_pts)
    no_of_pts = in_pts.shape[1]

    res = opt.minimize(getSSDShear, init_guess, args=(in_pts, out_pts, no_of_pts),
                       method=opt_method, bounds=opt_bounds, jac=jacobian_func)
    if debug_mode_sh:
        print 'Shear  Optimization done: '
        print 'x: ', res.x
        print 'success: ', res.success
        print 'message: ', res.message

    opt_shear_mat = getShearingMatrix(res.x[0], res.x[1])
    return opt_shear_mat

def computeProjOpt(in_pts, out_pts, opt_method, jacobian_func=None, v10=0, v20=0):
    max_v1 = 1.0
    min_v1 = -1.0
    max_v2 = 1.0
    min_v2 = -1.0

    opt_bounds = ((min_v1, max_v1), (min_v2, max_v2))
    init_guess = np.asarray([v10, v20])

    in_pts = np.mat(in_pts)
    out_pts = np.mat(out_pts)
    no_of_pts = in_pts.shape[1]

    res = opt.minimize(getSSDProj, init_guess, args=(in_pts, out_pts, no_of_pts),
                       method=opt_method, bounds=opt_bounds, jac=jacobian_func)
    if debug_mode_sh:
        print 'Proj  Optimization done: '
        print 'x: ', res.x
        print 'success: ', res.success
        print 'message: ', res.message

    opt_proj_mat = getProjectionMatrix(res.x[0], res.x[1])
    return opt_proj_mat

def computeRTOpt(in_pts, out_pts, opt_method, jacobian_func=None, tx0=0, ty0=0, theta0=0):
    min_trans = -500.0
    max_trans = 500.0
    min_theta0 = -math.pi / 2.0
    max_theta0 = math.pi / 2.0
    init_guess = np.asarray([tx0, ty0, theta0])

    no_of_pts = in_pts.shape[1]
    opt_bounds = ((min_trans, max_trans), (min_trans, max_trans), (min_theta0, max_theta0))

    res = opt.minimize(getSSDRT, init_guess, args=(in_pts, out_pts, no_of_pts),
                       method=opt_method, bounds=opt_bounds, jac=jacobian_func)
    if debug_mode:
        print 'RT Optimization done: '
        print 'x: ', res.x
        print 'success: ', res.success
        print 'message: ', res.message

    (trans_mat, rot_mat) = getDecomposedRTMatrices(res.x)
    rt_mat = trans_mat * rot_mat
    return rt_mat, trans_mat, rot_mat


def computeSE2Opt(in_pts, out_pts, opt_method, jacobian_func=None, tx0=0, ty0=0, theta0=0, scale0=0):
    min_trans = -500.0
    max_trans = 500.0
    min_scale = 0.0
    max_scale = 10.0
    min_theta0 = -math.pi / 2.0
    max_theta0 = math.pi / 2.0
    init_guess = np.asarray([tx0, ty0, theta0, scale0])

    no_of_pts = in_pts.shape[1]
    opt_bounds = ((min_trans, max_trans), (min_trans, max_trans), (min_theta0, max_theta0), (min_scale, max_scale))

    res = opt.minimize(getSSDSE2, init_guess, args=(in_pts, out_pts, no_of_pts),
                       method=opt_method, bounds=opt_bounds, jac=jacobian_func)
    if debug_mode:
        print 'SE2 Optimization done: '
        print 'x: ', res.x
        print 'success: ', res.success
        print 'message: ', res.message

    (trans_mat, rot_mat, scale_mat) = getDecomposedSE2Matrices(res.x)
    se2_mat = trans_mat * rot_mat * scale_mat
    return se2_mat, trans_mat, rot_mat, scale_mat

def computeTranscalingOpt(in_pts, out_pts, opt_method, jacobian_func=None, tx0=0, ty0=0, theta0=0, scale0=0):
    min_trans = -500.0
    max_trans = 500.0
    min_scale = 0.0
    max_scale = 10.0
    init_guess = np.asarray([tx0, ty0, scale0])

    no_of_pts = in_pts.shape[1]
    opt_bounds = ((min_trans, max_trans), (min_trans, max_trans), (min_scale, max_scale))

    res = opt.minimize(getSSDTranscaling, init_guess, args=(in_pts, out_pts, no_of_pts),
                       method=opt_method, bounds=opt_bounds, jac=jacobian_func)
    if debug_mode:
        print 'Transcaling Optimization done: '
        print 'x: ', res.x
        print 'success: ', res.success
        print 'message: ', res.message

    trs_mat = getTranscalingMatrix(res.x[0], res.x[1], res.x[2])
    return trs_mat


def computeAffineOpt(in_pts, out_pts, opt_method, jacobian_func=None, tx0=0, ty0=0,
                     theta0=0, scale0=0, a0=0, b0=0):
    min_trans = -500.0
    max_trans = 500.0
    min_scale = 0.0
    max_scale = 10.0
    min_theta0 = -math.pi / 2.0
    max_theta0 = math.pi / 2.0
    max_a = 10.0
    min_a = -10.0
    max_b = 10.0
    min_b = -10.0
    init_guess = np.asarray([tx0, ty0, theta0, scale0, a0, b0])

    no_of_pts = in_pts.shape[1]
    opt_bounds = ((min_trans, max_trans), (min_trans, max_trans), (min_theta0, max_theta0),
                  (min_scale, max_scale), (min_a, max_a), (min_b, max_b))

    res = opt.minimize(getSSDAffine, init_guess, args=(in_pts, out_pts, no_of_pts),
                       method=opt_method, bounds=opt_bounds, jac=jacobian_func)
    if debug_mode:
        print 'Affine Optimization done: '
        print 'x: ', res.x
        print 'success: ', res.success
        print 'message: ', res.message

    (trans_mat, rot_mat, scale_mat, shear_mat) = getDecomposedAffineMatrices(res.x)
    affine_mat = trans_mat * rot_mat * scale_mat * shear_mat
    return affine_mat, trans_mat, rot_mat, scale_mat, shear_mat