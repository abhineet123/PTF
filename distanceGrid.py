from decomposition import *
import CModules.distanceUtils as distanceUtils

invalid_dist = -1


def getHomDistanceGridPre(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals,
                          tx_vec, ty_vec, theta_vec, scale_vec, a_vec, b_vec, v1_vec, v2_vec,
                          dist_func=getSSDPoints):
    std_pts_hm = util.homogenize(std_pts)
    res = (tx_vec.shape[0], ty_vec.shape[0],
           theta_vec.shape[0], scale_vec.shape[0],
           a_vec.shape[0], b_vec.shape[0],
           v1_vec.shape[0], v2_vec.shape[0])

    dist_grid = np.zeros(res, dtype=np.float64)
    opt_params = np.zeros([8, 1])
    min_dist = np.inf

    tx_id = 0
    for tx in tx_vec:
        hom_mat = curr_hom_mat
        ty_id = 0
        for ty in ty_vec:
            trans_mat = np.mat(
                [[1, 0, tx],
                 [0, 1, ty],
                 [0, 0, 1]]
            )
            hom_mat = trans_mat * hom_mat
            theta_id = 0
            for theta in theta_vec:
                rot_mat = getRotationMatrix(theta)
                hom_mat = rot_mat * hom_mat
                scale_id = 0
                for scale in scale_vec:
                    scaling_mat = getRotationMatrix(scale)
                    hom_mat = scaling_mat * hom_mat
                    a_id = 0
                    for a in a_vec:
                        b_id = 0
                        for b in b_vec:
                            shear_mat = getShearingMatrix(a, b)
                            hom_mat = shear_mat * hom_mat
                            v1_id = 0
                            for v1 in v1_vec:
                                v2_id = 0
                                for v2 in v2_vec:
                                    proj_mat = getProjectionMatrix(v1, v2)
                                    hom_mat = proj_mat * hom_mat
                                    hom_pts = util.dehomogenize(curr_norm_mat * hom_mat * std_pts_hm)
                                    x = hom_pts[0, :]
                                    y = hom_pts[1, :]

                                    lx = np.floor(x).astype(np.uint16)
                                    ux = np.ceil(x).astype(np.uint16)
                                    ly = np.floor(y).astype(np.uint16)
                                    uy = np.ceil(y).astype(np.uint16)

                                    dx = x - lx
                                    dy = y - ly

                                    ll = np.multiply((1 - dx), (1 - dy))
                                    lu = np.multiply(dx, (1 - dy))
                                    ul = np.multiply((1 - dx), dy)
                                    uu = np.multiply(dx, dy)

                                    try:
                                        hom_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux],
                                                                                                    lu) + \
                                                         np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
                                    except IndexError:
                                        dist = invalid_dist
                                    else:
                                        dist = dist_func(hom_pixel_vals, ref_pixel_vals)
                                        if dist < min_dist:
                                            min_dist = dist
                                            opt_params = [tx, ty, theta, scale, a, b, v1, v2]
                                    dist_grid[tx_id, ty_id, theta_id, scale_id, a_id, b_id, v1_id, v2_id] = dist

                                    v2_id += 1
                                v1_id += 1
                            b_id += 1
                        a_id += 1
                    scale_id += 1
                theta_id += 1
            ty_id += 1
        tx_id += 1
    return opt_params, dist_grid


def getTransDistanceGridPre(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, tx_vec, ty_vec,
                            dist_func=getSSDPoints):
    std_pts_hm = util.homogenize(std_pts)
    res = (tx_vec.shape[0], ty_vec.shape[0])
    dist_grid = np.zeros(res, dtype=np.float64)

    post_mat = curr_norm_mat * curr_hom_mat
    row = 0
    for tx in tx_vec:
        col = 0
        for ty in ty_vec:
            trans_mat = np.mat(
                [[1, 0, tx],
                 [0, 1, ty],
                 [0, 0, 1]]
            )
            trans_pts = util.dehomogenize(post_mat * trans_mat * std_pts_hm)
            x = trans_pts[0, :]
            y = trans_pts[1, :]

            lx = np.floor(x).astype(np.uint16)
            ux = np.ceil(x).astype(np.uint16)
            ly = np.floor(y).astype(np.uint16)
            uy = np.ceil(y).astype(np.uint16)

            dx = x - lx
            dy = y - ly

            ll = np.multiply((1 - dx), (1 - dy))
            lu = np.multiply(dx, (1 - dy))
            ul = np.multiply((1 - dx), dy)
            uu = np.multiply(dx, dy)

            try:
                trans_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                   np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
                # np.savetxt('trans_pixel_vals.txt', trans_pixel_vals, fmt='%9.5f')
            except IndexError:
                # print 'Invalid coordinates obtained with tx: ', tx, ' and ty: ', ty
                # print 'max trans_pts:\n', np.max(trans_pts, axis=1)
                # print 'min trans_pts:\n', np.min(trans_pts, axis=1)
                # np.savetxt('trans_pts.txt', trans_pts, fmt='%10.5f')
                # print 'trans_pts:\n', trans_pts
                dist_grid[row, col] = invalid_dist
            else:
                dist_grid[row, col] = dist_func(trans_pixel_vals, ref_pixel_vals)
                # print 'dist: ', dist_grid[row, col]
            col += 1
        row += 1
    return dist_grid


def getTransDistanceGridPost(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, tx_vec, ty_vec,
                             dist_func=getSSDPoints):
    std_pts_hm = util.homogenize(std_pts)
    res = (tx_vec.shape[0], ty_vec.shape[0])
    dist_grid = np.zeros(res, dtype=np.float64)

    post_mat = curr_norm_mat * curr_hom_mat
    trans_pts = util.dehomogenize(post_mat * std_pts_hm)
    row = 0
    for tx in tx_vec:
        col = 0
        for ty in ty_vec:
            x = trans_pts[0, :] + tx
            y = trans_pts[1, :] + ty

            lx = np.floor(x).astype(np.uint16)
            ux = np.ceil(x).astype(np.uint16)
            ly = np.floor(y).astype(np.uint16)
            uy = np.ceil(y).astype(np.uint16)

            dx = x - lx
            dy = y - ly

            ll = np.multiply((1 - dx), (1 - dy))
            lu = np.multiply(dx, (1 - dy))
            ul = np.multiply((1 - dx), dy)
            uu = np.multiply(dx, dy)

            try:
                trans_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                   np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
            except IndexError:
                # print 'Invalid coordinates obtained with tx: ', tx, ' and ty: ', ty
                # print 'max trans_pts:\n', np.max(trans_pts, axis=1)
                # print 'min trans_pts:\n', np.min(trans_pts, axis=1)
                # np.savetxt('trans_pts.txt', trans_pts, fmt='%10.5f')
                # print 'trans_pts:\n', trans_pts
                dist_grid[row, col] = invalid_dist
            else:
                dist_grid[row, col] = dist_func(trans_pixel_vals, ref_pixel_vals)
            col += 1
        row += 1
    return dist_grid


# ------------------------------ Compositional ------------------------------#


def getRTxDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, theta_vec, tx_vec,
                           dist_func=getSSDPoints):
    std_pts_hm = util.homogenize(std_pts)
    n_pts = std_pts.shape[1]
    res = (theta_vec.shape[0], tx_vec.shape[0])
    rtx_dist_grid = np.zeros(res, dtype=np.float64)

    post_mat = curr_norm_mat * curr_hom_mat

    row = 0
    for theta in theta_vec:

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rot_mat = np.mat(
            [[cos_theta, - sin_theta, 0],
             [sin_theta, cos_theta, 0],
             [0, 0, 1]]
        )
        post_rot_mat = post_mat * rot_mat
        col = 0
        for tx in tx_vec:
            trans_mat = np.mat(
                [[1, 0, tx],
                 [0, 1, 0],
                 [0, 0, 1]]
            )
            rtx_pts = util.dehomogenize(post_rot_mat * trans_mat * std_pts_hm)
            x = rtx_pts[0, :]
            y = rtx_pts[1, :]

            lx = np.floor(x).astype(np.uint16)
            ux = np.ceil(x).astype(np.uint16)
            ly = np.floor(y).astype(np.uint16)
            uy = np.ceil(y).astype(np.uint16)

            dx = x - lx
            dy = y - ly

            ll = np.multiply((1 - dx), (1 - dy))
            lu = np.multiply(dx, (1 - dy))
            ul = np.multiply((1 - dx), dy)
            uu = np.multiply(dx, dy)

            try:
                rtx_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                 np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
                # rtx_dist_grid[row, col] = np.sum(np.square(rtx_pixel_vals - ref_pixel_vals))
            except IndexError:
                # print 'Invalid coordinates obtained with scale: ', scale, ' and theta: ', theta
                rtx_dist_grid[row, col] = invalid_dist
            else:
                rtx_dist_grid[row, col] = dist_func(rtx_pixel_vals, ref_pixel_vals)

            col += 1
        row += 1
    return rtx_dist_grid / n_pts


def getRTyDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, theta_vec, ty_vec,
                           dist_func=getSSDPoints):
    std_pts_hm = util.homogenize(std_pts)
    n_pts = std_pts.shape[1]
    res = (theta_vec.shape[0], ty_vec.shape[0])
    rty_dist_grid = np.zeros(res, dtype=np.float64)

    post_mat = curr_norm_mat * curr_hom_mat

    row = 0
    for theta in theta_vec:

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rot_mat = np.mat(
            [[cos_theta, - sin_theta, 0],
             [sin_theta, cos_theta, 0],
             [0, 0, 1]]
        )
        post_rot_mat = post_mat * rot_mat
        col = 0
        for ty in ty_vec:
            trans_mat = np.mat(
                [[1, 0, 0],
                 [0, 1, ty],
                 [0, 0, 1]]
            )

            rty_pts = util.dehomogenize(post_rot_mat * trans_mat * std_pts_hm)
            x = rty_pts[0, :]
            y = rty_pts[1, :]

            lx = np.floor(x).astype(np.uint16)
            ux = np.ceil(x).astype(np.uint16)
            ly = np.floor(y).astype(np.uint16)
            uy = np.ceil(y).astype(np.uint16)

            dx = x - lx
            dy = y - ly

            ll = np.multiply((1 - dx), (1 - dy))
            lu = np.multiply(dx, (1 - dy))
            ul = np.multiply((1 - dx), dy)
            uu = np.multiply(dx, dy)

            try:
                rty_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                 np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
                # rty_dist_grid[row, col] = np.sum(np.square(rty_pixel_vals - ref_pixel_vals))
            except IndexError:
                # print 'Invalid coordinates obtained with scale: ', scale, ' and theta: ', theta
                rty_dist_grid[row, col] = invalid_dist
            else:
                rty_dist_grid[row, col] = dist_func(rty_pixel_vals, ref_pixel_vals)

            col += 1
        row += 1
    return rty_dist_grid / n_pts


def getRSDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, scale_vec, theta_vec,
                          dist_func=getSSDPoints):
    std_pts_hm = util.homogenize(std_pts)
    n_pts = std_pts.shape[1]
    res = (scale_vec.shape[0], theta_vec.shape[0])
    rs_dist_grid = np.zeros(res, dtype=np.float64)

    post_mat = curr_norm_mat * curr_hom_mat

    row = 0
    for scale in scale_vec:
        col = 0
        scale_mat = getScalingMatrix(scale)
        post_scale_mat = post_mat * scale_mat
        # print 'scale: ', scale
        # print 'scale_mat:\n', scale_mat
        for theta in theta_vec:

            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            rot_mat = np.mat(
                [[cos_theta, - sin_theta, 0],
                 [sin_theta, cos_theta, 0],
                 [0, 0, 1]]
            )
            # rot_mat = getRotationMatrix(theta)

            rs_pts = util.dehomogenize(post_scale_mat * rot_mat * std_pts_hm)

            # rs_pixel_vals = np.mat([util.bilin_interp(img, rs_pts[0, pt_id], rs_pts[1, pt_id]) for pt_id in
            # xrange(n_pts)])

            x = rs_pts[0, :]
            y = rs_pts[1, :]

            lx = np.floor(x).astype(np.uint16)
            ux = np.ceil(x).astype(np.uint16)
            ly = np.floor(y).astype(np.uint16)
            uy = np.ceil(y).astype(np.uint16)

            # if np.any(np.logical_or(lx > w, lx < 0)) or np.any(np.logical_or(ux > w, ux < 0)) or np.any(
            # np.logical_or(ly > h, ly < 0)) or np.any(np.logical_or(uy > h, uy < 0)):
            # raise SyntaxError('Invalid indices obtained with scale: ', scale, ' and theta: ', theta)

            # lx[lx > w] = w
            # ux[ux > w] = w
            # ly[ly > h] = h
            # uy[uy > h] = h

            dx = x - lx
            dy = y - ly

            ll = np.multiply((1 - dx), (1 - dy))
            lu = np.multiply(dx, (1 - dy))
            ul = np.multiply((1 - dx), dy)
            uu = np.multiply(dx, dy)

            try:
                rs_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
                # rs_dist_grid[row, col] = np.sum(np.square(rs_pixel_vals - ref_pixel_vals))
            except IndexError:
                # print 'Invalid coordinates obtained with scale: ', scale, ' and theta: ', theta
                rs_dist_grid[row, col] = invalid_dist
            else:
                rs_dist_grid[row, col] = dist_func(rs_pixel_vals, ref_pixel_vals)

            # np.savetxt('pts_hm.txt', pts_hm, fmt='%10.5f')
            # np.savetxt('rs_pts.txt', rs_pts, fmt='%10.5f')
            # np.savetxt('rs_pixel_vals.txt', rs_pixel_vals, fmt='%10.5f')
            # raw_input('Press any key to continue')
            # print 'rs_pts:\n', rs_pts
            # print 'rot_mat:\n', rot_mat
            # print 'dist:', dist
            # print 'tx: ', tx, 'ty: ', ty
            col += 1
        row += 1
    return rs_dist_grid / n_pts


def getRSDistanceGridComp2(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, scale_vec, theta_vec,
                           dist_func=getSSDPoints):
    std_pts_hm = util.homogenize(std_pts)
    n_pts = std_pts.shape[1]
    res = (theta_vec.shape[0], scale_vec.shape[0])
    rs_dist_grid = np.zeros(res, dtype=np.float64)
    post_mat = curr_norm_mat * curr_hom_mat
    x = np.zeros((theta_vec.shape[0], n_pts))
    y = np.zeros((theta_vec.shape[0], n_pts))
    ref_pixel_vals_mat = np.tile(ref_pixel_vals, (theta_vec.shape[0], 1))

    col = 0
    for scale in scale_vec:
        row = 0
        scale_mat = getScalingMatrix(scale)
        post_scale_mat = post_mat * scale_mat
        for theta in theta_vec:
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            rot_mat = np.mat(
                [[cos_theta, - sin_theta, 0],
                 [sin_theta, cos_theta, 0],
                 [0, 0, 1]]
            )
            rs_pts = util.dehomogenize(post_scale_mat * rot_mat * std_pts_hm)
            x[row, :] = rs_pts[0, :]
            y[row, :] = rs_pts[1, :]
            row += 1
        lx = np.floor(x).astype(np.uint16)
        ux = np.ceil(x).astype(np.uint16)
        ly = np.floor(y).astype(np.uint16)
        uy = np.ceil(y).astype(np.uint16)

        dx = x - lx
        dy = y - ly

        ll = np.multiply((1 - dx), (1 - dy))
        lu = np.multiply(dx, (1 - dy))
        ul = np.multiply((1 - dx), dy)
        uu = np.multiply(dx, dy)

        try:
            rs_pixel_vals_mat = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
        except IndexError:
            rs_dist_grid[:, col] = invalid_dist
        else:
            rs_dist_grid[:, col] = dist_func(rs_pixel_vals_mat, ref_pixel_vals_mat)

        col += 1
    return rs_dist_grid.transpose() / n_pts


def getShearDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, a_vec, b_vec,
                             dist_func=getSSDPoints):
    std_pts_hm = util.homogenize(std_pts)
    n_pts = std_pts.shape[1]
    res = (a_vec.shape[0], b_vec.shape[0])
    shear_dist_grid = np.zeros(res, dtype=np.float64)

    post_mat = curr_norm_mat * curr_hom_mat

    row = 0
    for a in a_vec:
        col = 0
        for b in b_vec:

            shear_mat = np.mat(
                [[1 + a, b, 0],
                 [0, 1, 0],
                 [0, 0, 1]]
            )
            # rot_mat = getRotationMatrix(theta)

            shear_pts = util.dehomogenize(post_mat * shear_mat * std_pts_hm)

            x = shear_pts[0, :]
            y = shear_pts[1, :]

            lx = np.floor(x).astype(np.uint16)
            ux = np.ceil(x).astype(np.uint16)
            ly = np.floor(y).astype(np.uint16)
            uy = np.ceil(y).astype(np.uint16)

            dx = x - lx
            dy = y - ly

            ll = np.multiply((1 - dx), (1 - dy))
            lu = np.multiply(dx, (1 - dy))
            ul = np.multiply((1 - dx), dy)
            uu = np.multiply(dx, dy)

            try:
                shear_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                   np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
            except IndexError:
                # print 'Invalid coordinates obtained with da: ', da, ' and db: ', db
                shear_dist_grid[row, col] = invalid_dist
            else:
                shear_dist_grid[row, col] = dist_func(shear_pixel_vals, ref_pixel_vals)

            col += 1
        row += 1
    return shear_dist_grid / n_pts


def getProjDistanceGridComp(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, v1_vec, v2_vec,
                            dist_func=getSSDPoints):
    std_pts_hm = util.homogenize(std_pts)
    n_pts = std_pts.shape[1]
    res = (v1_vec.shape[0], v2_vec.shape[0])
    proj_dist_grid = np.zeros(res, dtype=np.float64)

    post_mat = curr_norm_mat * curr_hom_mat

    row = 0
    for v1 in v1_vec:
        col = 0
        for v2 in v2_vec:

            proj_mat = np.mat(
                [[1, 0, 0],
                 [0, 1, 0],
                 [v1, v2, 1]]
            )
            proj_pts = util.dehomogenize(post_mat * proj_mat * std_pts_hm)

            x = proj_pts[0, :]
            y = proj_pts[1, :]

            lx = np.floor(x).astype(np.uint16)
            ux = np.ceil(x).astype(np.uint16)
            ly = np.floor(y).astype(np.uint16)
            uy = np.ceil(y).astype(np.uint16)

            dx = x - lx
            dy = y - ly

            ll = np.multiply((1 - dx), (1 - dy))
            lu = np.multiply(dx, (1 - dy))
            ul = np.multiply((1 - dx), dy)
            uu = np.multiply(dx, dy)

            try:
                rs_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
            except IndexError:
                # print 'Invalid coordinates obtained with v1: ', v1, ' and v2: ', v2
                proj_dist_grid[row, col] = invalid_dist
            else:
                proj_dist_grid[row, col] = dist_func(rs_pixel_vals, ref_pixel_vals)

            col += 1
        row += 1
    return proj_dist_grid / n_pts


# ------------------------------ Additive ------------------------------#

def getRTxDistanceGridAdd(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, dtheta_vec, dtx_vec,
                          dist_func=getSSDPoints):
    affine_mat, proj_mat, hom_rec_mat = decomposeHomographyInverse(curr_hom_mat)
    trans_mat, rt_mat, se2_mat, affine_rec_mat, params = decomposeAffineInverse(affine_mat)
    tx = params[0]
    ty = params[1]
    theta = params[2]
    scale = params[3]
    a = params[4]
    b = params[5]
    rec_trans_mat = getTranslationMatrix(0, ty)
    rec_scale_mat = getScalingMatrix(scale)
    rec_shear_mat = getShearingMatrix(a, b)

    std_pts_hm = util.homogenize(std_pts)
    std_pts_trans_hm = rec_trans_mat * std_pts_hm
    post_mat = curr_norm_mat * proj_mat * rec_shear_mat * rec_scale_mat

    n_pts = std_pts.shape[1]
    res = (dtheta_vec.shape[0], dtx_vec.shape[0])
    rtx_dist_grid = np.zeros(res, dtype=np.float64)

    row = 0
    for dtheta in dtheta_vec:

        cos_theta = math.cos(theta + dtheta)
        sin_theta = math.sin(theta + dtheta)
        rot_mat = np.mat(
            [[cos_theta, - sin_theta, 0],
             [sin_theta, cos_theta, 0],
             [0, 0, 1]]
        )
        col = 0
        post_rot_mat = post_mat * rot_mat
        for dtx in dtx_vec:
            trans_mat = np.mat(
                [[1, 0, tx + dtx],
                 [0, 1, 0],
                 [0, 0, 1]]
            )
            rtx_pts = util.dehomogenize(post_rot_mat * trans_mat * std_pts_trans_hm)

            x = rtx_pts[0, :]
            y = rtx_pts[1, :]

            lx = np.floor(x).astype(np.uint16)
            ux = np.ceil(x).astype(np.uint16)
            ly = np.floor(y).astype(np.uint16)
            uy = np.ceil(y).astype(np.uint16)

            dx = x - lx
            dy = y - ly

            ll = np.multiply((1 - dx), (1 - dy))
            lu = np.multiply(dx, (1 - dy))
            ul = np.multiply((1 - dx), dy)
            uu = np.multiply(dx, dy)

            try:
                rtx_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                 np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
            except IndexError:
                # print 'Invalid coordinates obtained with scale: ', scale, ' and theta: ', theta
                rtx_dist_grid[row, col] = invalid_dist
            else:
                rtx_dist_grid[row, col] = dist_func(rtx_pixel_vals, ref_pixel_vals)
            col += 1
        row += 1
    return rtx_dist_grid / n_pts


def getRTyDistanceGridAdd(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, dtheta_vec, dty_vec,
                          dist_func=getSSDPoints):
    affine_mat, proj_mat, hom_rec_mat = decomposeHomographyInverse(curr_hom_mat)
    trans_mat, rt_mat, se2_mat, affine_rec_mat, params = decomposeAffineInverse(affine_mat)
    tx = params[0]
    ty = params[1]
    theta = params[2]
    scale = params[3]
    a = params[4]
    b = params[5]
    rec_trans_mat = getTranslationMatrix(tx, 0)
    rec_scale_mat = getScalingMatrix(scale)
    rec_shear_mat = getShearingMatrix(a, b)

    std_pts_hm = util.homogenize(std_pts)
    std_pts_trans_hm = rec_trans_mat * std_pts_hm
    post_mat = curr_norm_mat * proj_mat * rec_shear_mat * rec_scale_mat

    n_pts = std_pts.shape[1]
    res = (dtheta_vec.shape[0], dty_vec.shape[0])
    rty_dist_grid = np.zeros(res, dtype=np.float64)

    row = 0
    for dtheta in dtheta_vec:

        cos_theta = math.cos(theta + dtheta)
        sin_theta = math.sin(theta + dtheta)
        rot_mat = np.mat(
            [[cos_theta, - sin_theta, 0],
             [sin_theta, cos_theta, 0],
             [0, 0, 1]]
        )
        col = 0
        post_rot_mat = post_mat * rot_mat
        for dty in dty_vec:
            trans_mat = np.mat(
                [[1, 0, 0],
                 [0, 1, ty + dty],
                 [0, 0, 1]]
            )
            rty_pts = util.dehomogenize(post_rot_mat * trans_mat * std_pts_trans_hm)

            x = rty_pts[0, :]
            y = rty_pts[1, :]

            lx = np.floor(x).astype(np.uint16)
            ux = np.ceil(x).astype(np.uint16)
            ly = np.floor(y).astype(np.uint16)
            uy = np.ceil(y).astype(np.uint16)

            dx = x - lx
            dy = y - ly

            ll = np.multiply((1 - dx), (1 - dy))
            lu = np.multiply(dx, (1 - dy))
            ul = np.multiply((1 - dx), dy)
            uu = np.multiply(dx, dy)

            try:
                rty_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                 np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
            except IndexError:
                # print 'Invalid coordinates obtained with scale: ', scale, ' and theta: ', theta
                rty_dist_grid[row, col] = invalid_dist
            else:
                rty_dist_grid[row, col] = dist_func(rty_pixel_vals, ref_pixel_vals)
            col += 1
        row += 1
    return rty_dist_grid / n_pts


def getRSDistanceGridAdd(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, dscale_vec, dtheta_vec,
                         dist_func=getSSDPoints):
    affine_mat, proj_mat, hom_rec_mat = decomposeHomographyInverse(curr_hom_mat)
    trans_mat, rt_mat, se2_mat, affine_rec_mat, params = decomposeAffineInverse(affine_mat)

    # print 'dscale_vec: ', dscale_vec
    # print 'dtheta_vec: ', dtheta_vec

    theta = params[2]
    scale = params[3]

    a = params[4]
    b = params[5]
    rec_shear_mat = getShearingMatrix(a, b)
    # print 'getRSDistanceGridAdd: rec_shear_mat:\n', rec_shear_mat

    std_pts_hm = util.homogenize(std_pts)
    std_pts_trans_hm = trans_mat * std_pts_hm

    n_pts = std_pts.shape[1]
    res = (dscale_vec.shape[0], dtheta_vec.shape[0])
    rs_dist_grid = np.zeros(res, dtype=np.float64)

    post_mat = curr_norm_mat * proj_mat * rec_shear_mat

    row = 0
    for dscale in dscale_vec:
        col = 0
        scale_mat = getScalingMatrix(scale + dscale)
        post_scale_mat = post_mat * scale_mat

        # print 'scale_mat:\n', scale_mat
        for dtheta in dtheta_vec:

            cos_theta = math.cos(theta + dtheta)
            sin_theta = math.sin(theta + dtheta)
            rot_mat = np.mat(
                [[cos_theta, - sin_theta, 0],
                 [sin_theta, cos_theta, 0],
                 [0, 0, 1]]
            )
            # rot_mat = getRotationMatrix(theta + dtheta)
            # print 'dscale: ', dscale
            # print 'dtheta: ', dtheta
            # print 'rot_mat:\n', rot_mat
            # rec_se2_mat = scale_mat * rot_mat * trans_mat
            # rec_affine_mat = rec_shear_mat * rec_se2_mat
            # rec_hom_mat = post_scale_mat * rot_mat * trans_mat

            # print 'rec_se2_mat:\n', rec_se2_mat
            # print 'se2_mat:\n', se2_mat
            #
            # print 'rec_affine_mat:\n', rec_affine_mat
            # print 'affine_rec_mat:\n', affine_rec_mat
            # print 'affine_mat:\n', affine_mat
            #
            # print 'rec_hom_mat:\n', rec_hom_mat
            # print 'hom_rec_mat:\n', hom_rec_mat
            # print 'curr_hom_mat:\n', curr_hom_mat

            rs_pts = util.dehomogenize(post_scale_mat * rot_mat * std_pts_trans_hm)

            x = rs_pts[0, :]
            y = rs_pts[1, :]

            lx = np.floor(x).astype(np.uint16)
            ux = np.ceil(x).astype(np.uint16)
            ly = np.floor(y).astype(np.uint16)
            uy = np.ceil(y).astype(np.uint16)

            # lx[lx >= w] = w
            # ux[ux >= w] = w
            # ly[ly >= h] = h
            # uy[uy >= h] = h

            # if np.any(np.logical_or(lx >= w, lx < 0)) or np.any(np.logical_or(ux >= w, ux < 0)) or np.any(
            # np.logical_or(ly >= h, ly < 0)) or np.any(np.logical_or(uy >= h, uy < 0)):
            # raise SyntaxError('Invalid indices obtained with scale: ', scale, ' and theta: ', theta)

            dx = x - lx
            dy = y - ly

            ll = np.multiply((1 - dx), (1 - dy))
            lu = np.multiply(dx, (1 - dy))
            ul = np.multiply((1 - dx), dy)
            uu = np.multiply(dx, dy)

            try:
                rs_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
            except IndexError:
                # print 'Invalid coordinates obtained with scale: ', scale, ' and theta: ', theta
                rs_dist_grid[row, col] = invalid_dist
            else:
                rs_dist_grid[row, col] = dist_func(rs_pixel_vals, ref_pixel_vals)



            # rs_pixel_vals = np.mat([util.bilin_interp(img, rs_pts[0, pt_id], rs_pts[1, pt_id]) for pt_id in
            # xrange(n_pts)])

            # np.savetxt('pts_hm.txt', pts_hm, fmt='%10.5f')
            # np.savetxt('rs_pts.txt', rs_pts, fmt='%10.5f')
            # np.savetxt('rs_pixel_vals.txt', rs_pixel_vals, fmt='%10.5f')
            # raw_input('Press any key to continue')

            # print 'rs_pts:\n', rs_pts
            # print 'rot_mat:\n', rot_mat
            # print 'dist:', dist
            # print 'tx: ', tx, 'ty: ', ty
            col += 1
        row += 1
    return rs_dist_grid / n_pts


def getShearDistanceGridAdd(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, da_vec, db_vec,
                            dist_func=getSSDPoints):
    affine_mat, proj_mat, hom_rec_mat = decomposeHomographyInverse(curr_hom_mat)
    trans_mat, rt_mat, se2_mat, affine_rec_mat, params = decomposeAffineInverse(affine_mat)
    a = params[4]
    b = params[5]

    std_pts_hm = util.homogenize(std_pts)
    std_pts_se2_hm = se2_mat * std_pts_hm

    res = (da_vec.shape[0], db_vec.shape[0])
    shear_dist_grid = np.zeros(res, dtype=np.float64)

    post_mat = curr_norm_mat * proj_mat

    row = 0
    for da in da_vec:
        col = 0
        for db in db_vec:
            shear_mat = np.mat(
                [[1 + a + da, b + db, 0],
                 [0, 1, 0],
                 [0, 0, 1]]
            )

            shear_pts = util.dehomogenize(post_mat * shear_mat * std_pts_se2_hm)

            x = shear_pts[0, :]
            y = shear_pts[1, :]

            lx = np.floor(x).astype(np.uint16)
            ux = np.ceil(x).astype(np.uint16)
            ly = np.floor(y).astype(np.uint16)
            uy = np.ceil(y).astype(np.uint16)

            dx = x - lx
            dy = y - ly

            ll = np.multiply((1 - dx), (1 - dy))
            lu = np.multiply(dx, (1 - dy))
            ul = np.multiply((1 - dx), dy)
            uu = np.multiply(dx, dy)

            try:
                shear_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                   np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
            except IndexError:
                # print 'Invalid coordinates obtained with da: ', da, ' and db: ', db
                shear_dist_grid[row, col] = invalid_dist
            else:
                shear_dist_grid[row, col] = dist_func(shear_pixel_vals, ref_pixel_vals)

            col += 1
        row += 1
    return shear_dist_grid / std_pts.shape[1]


def getProjDistanceGridAdd(std_pts, curr_hom_mat, curr_norm_mat, img, ref_pixel_vals, dv1_vec, dv2_vec,
                           dist_func=getSSDPoints):
    affine_mat, proj_mat, hom_rec_mat = decomposeHomographyInverse(curr_hom_mat)
    v1 = proj_mat[2, 0]
    v2 = proj_mat[2, 1]

    std_pts_hm = util.homogenize(std_pts)
    std_pts_affine_hm = affine_mat * std_pts_hm

    res = (dv1_vec.shape[0], dv2_vec.shape[0])
    proj_dist_grid = np.zeros(res, dtype=np.float64)

    row = 0
    for dv1 in dv1_vec:
        col = 0
        for dv2 in dv2_vec:
            proj_mat = np.mat(
                [[1, 0, 0],
                 [0, 1, 0],
                 [v1 + dv1, v2 + dv2, 1]]
            )

            proj_pts = util.dehomogenize(curr_norm_mat * proj_mat * std_pts_affine_hm)

            x = proj_pts[0, :]
            y = proj_pts[1, :]

            lx = np.floor(x).astype(np.uint16)
            ux = np.ceil(x).astype(np.uint16)
            ly = np.floor(y).astype(np.uint16)
            uy = np.ceil(y).astype(np.uint16)

            dx = x - lx
            dy = y - ly

            ll = np.multiply((1 - dx), (1 - dy))
            lu = np.multiply(dx, (1 - dy))
            ul = np.multiply((1 - dx), dy)
            uu = np.multiply(dx, dy)

            try:
                proj_pixel_vals = np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
                                  np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)
            except IndexError:
                # print 'Invalid coordinates obtained with da: ', da, ' and db: ', db
                proj_dist_grid[row, col] = invalid_dist
            else:
                proj_dist_grid[row, col] = dist_func(proj_pixel_vals, ref_pixel_vals)

            col += 1
        row += 1
    return proj_dist_grid / std_pts.shape[1]


def getIndexOfMinimum(dist_grid):
    dist_grid[dist_grid < 0] = float("inf")
    min_id = np.argmin(dist_grid)
    row_id = int(min_id / dist_grid.shape[1])
    col_id = int(min_id % dist_grid.shape[1])
    return row_id, col_id


def getIndexOfMaximum(dist_grid):
    max_id = np.argmax(dist_grid)
    row_id = int(max_id / dist_grid.shape[1])
    col_id = int(max_id % dist_grid.shape[1])
    return row_id, col_id


def getDistanceFunction(appearance_model, resy, resx, n_bins):
    pre_proc_func = lambda img: img * (n_bins - 1) / 255.0
    post_proc_func = lambda pixel_vals: np.rint(pixel_vals)
    opt_func = getIndexOfMaximum
    is_better = lambda x, y : x > y

    if appearance_model == 'ssd':
        dist_func = getSSDPoints
        pre_proc_func = lambda img: img
        post_proc_func = lambda pixel_vals: pixel_vals
        opt_func = getIndexOfMinimum
        is_better = lambda x, y : x < y
    elif appearance_model == 'ssim':
        dist_func = getSSIMPoints
        pre_proc_func = lambda img: img
        post_proc_func = lambda pixel_vals: pixel_vals
        opt_func = getIndexOfMaximum
        is_better = lambda x, y : x > y
    elif appearance_model == 'mi':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getMIPoints
    elif appearance_model == 'bmi':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getBSplineMIPoints
        post_proc_func = lambda pixel_vals: pixel_vals
    elif appearance_model == 'mi2':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getMIPoints2
    elif appearance_model == 'mi_old':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getMIPointsOld
    elif appearance_model == 'ncc':
        dist_func = getNCCPoints
        pre_proc_func = lambda img: img - np.mean(img)
        post_proc_func = lambda pixel_vals: pixel_vals
    elif appearance_model == 'ncc2':
        dist_func = getNCCPoints2
        post_proc_func = lambda pixel_vals: pixel_vals
    elif appearance_model == 'scv':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getSCVPoints
        opt_func = getIndexOfMinimum
        is_better = lambda x, y : x < y
    elif appearance_model == 'scv2':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getSCVPoints2
        opt_func = getIndexOfMinimum
        is_better = lambda x, y : x < y
    elif appearance_model == 'ccre':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getCCREPoints
    elif appearance_model == 'hssd':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getHistSSDPoints
        opt_func = getIndexOfMinimum
        is_better = lambda x, y : x < y
    elif appearance_model == 'jht':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getJointHistTracePoints
    elif appearance_model == 'mssd':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getMIMatSSDPoints
        opt_func = getIndexOfMinimum
        is_better = lambda x, y: x < y
    elif appearance_model == 'bmssd':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getBSplineMIMatSSDPoints
        opt_func = getIndexOfMinimum
        is_better = lambda x, y: x < y
        post_proc_func = lambda pixel_vals: pixel_vals
    elif appearance_model == 'crv':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getCorrelationVariancePoints
        opt_func = getIndexOfMinimum
        is_better = lambda x, y: x < y
        post_proc_func = lambda pixel_vals: pixel_vals
    elif appearance_model == 'fkld':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getBSplineFKLDPoints
        opt_func = getIndexOfMinimum
        post_proc_func = lambda pixel_vals: pixel_vals
        is_better = lambda x, y: x < y
    elif appearance_model == 'ikld':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getBSplineIKLDPoints
        opt_func = getIndexOfMinimum
        post_proc_func = lambda pixel_vals: pixel_vals
        is_better = lambda x, y: x < y
    elif appearance_model == 'mkld':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getBSplineMKLDPoints
        opt_func = getIndexOfMinimum
        post_proc_func = lambda pixel_vals: pixel_vals
        is_better = lambda x, y: x < y
    elif appearance_model == 'chis':
        distanceUtils.initStateVars(resy, resx, n_bins)
        dist_func = distanceUtils.getBSplineCHISPoints
        opt_func = getIndexOfMinimum
        post_proc_func = lambda pixel_vals: pixel_vals
        is_better = lambda x, y: x < y
    else:
        raise StandardError('Invalid appearance model: {:s}'.format(appearance_model))

    if distanceUtils.isInited():
        print 'using distanceUtils module'

    return dist_func, pre_proc_func, post_proc_func, opt_func, is_better


def getGridVectors(vec_min, vec_max, vec_res, norm_factor=10):
    tx_vec = np.linspace(vec_min[0], vec_max[0], vec_res[0])
    ty_vec = np.linspace(vec_min[1], vec_max[1], vec_res[1])
    tx2_vec = np.linspace(vec_min[0] * norm_factor, vec_max[0] * norm_factor, vec_res[0])
    ty2_vec = np.linspace(vec_min[1] * norm_factor, vec_max[1] * norm_factor, vec_res[1])
    theta_vec = np.linspace(vec_min[2], vec_max[2], vec_res[2])
    scale_vec = np.linspace(vec_min[3], vec_max[3], vec_res[3])
    a_vec = np.linspace(vec_min[4], vec_max[4], vec_res[4])
    b_vec = np.linspace(vec_min[5], vec_max[5], vec_res[5])
    v1_vec = np.linspace(vec_min[6], vec_max[6], vec_res[6])
    v2_vec = np.linspace(vec_min[7], vec_max[7], vec_res[7])

    if np.count_nonzero(tx_vec) == vec_res[0]:
        tx_vec = np.insert(tx_vec, np.argwhere(tx_vec >= 0)[0, 0], 0).astype(np.float64)
    if np.count_nonzero(ty_vec) == vec_res[1]:
        ty_vec = np.insert(ty_vec, np.argwhere(ty_vec >= 0)[0, 0], 0).astype(np.float64)
    if np.count_nonzero(theta_vec) == vec_res[2]:
        theta_vec = np.insert(theta_vec, np.argwhere(theta_vec >= 0)[0, 0], 0).astype(np.float64)
    if np.count_nonzero(scale_vec) == vec_res[3]:
        scale_vec = np.insert(scale_vec, np.argwhere(scale_vec >= 0)[0, 0], 0).astype(np.float64)
    if np.count_nonzero(a_vec) == vec_res[4]:
        a_vec = np.insert(a_vec, np.argwhere(a_vec >= 0)[0, 0], 0).astype(np.float64)
    if np.count_nonzero(b_vec) == vec_res[5]:
        b_vec = np.insert(b_vec, np.argwhere(b_vec >= 0)[0, 0], 0).astype(np.float64)
    if np.count_nonzero(v1_vec) == vec_res[6]:
        v1_vec = np.insert(v1_vec, np.argwhere(v1_vec >= 0)[0, 0], 0).astype(np.float64)
    if np.count_nonzero(v2_vec) == vec_res[7]:
        v2_vec = np.insert(v2_vec, np.argwhere(v2_vec >= 0)[0, 0], 0).astype(np.float64)

    return tx_vec, ty_vec, theta_vec, scale_vec, a_vec, b_vec, v1_vec, v2_vec, tx2_vec, ty2_vec