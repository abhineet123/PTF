# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
This module contains highly optimized utility functions that are used in
the various tracking algorithms.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import cv2
import numpy as np
from scipy.linalg import expm
import pdb

cdef extern from "math.h":
    double floor(double) 
    double ceil(double)
    double sqrt(double)

# ---------- Homography Parameterizations ---------- #

cpdef double[:,:] make_hom_sl3(double[:] p) except *:
    cdef double[:,:] A = np.empty([3,3], dtype=np.float64)
    A[0,0] = p[0]
    A[0,1] = p[1]
    A[0,2] = p[2]
    A[1,0] = p[3]
    A[1,1] = p[4] - p[0]
    A[1,2] = p[5]
    A[2,0] = p[6]
    A[2,1] = p[7]
    A[2,2] = -p[4]
    return expm(np.mat(A))


cpdef double[:,:] aff_update_backward(double[:,:] warp, double[:] update) except *:
    cdef double[:,:] interm1 = np.empty([3,3], dtype=np.float64) 
    cdef double[:,:] interm2 = np.empty([3,3], dtype=np.float64)
    cdef double[:,:] ma1, ma2
    cdef double[:,:] res# = np.empty([1,6], dtype=np.float64)
    cdef double b
    cdef int i,j
    ma2 = np.empty([1,6], dtype=np.float64)
    ma2[0,0] = update[0]
    ma2[0,1] = update[3]
    ma2[0,2] = update[1]
    ma2[0,3] = update[4]
    ma2[0,4] = update[2]
    ma2[0,5] = update[5]
    # H inv
    b = 1.0 / ((1+ma2[0,0])*(1+ma2[0,3])-ma2[0,1]*ma2[0,2])
    # Construct the second term
    interm2[0,0] = 1 + b*(-ma2[0,0]-ma2[0,0]*ma2[0,3]+ma2[0,1]*ma2[0,2])
    interm2[1,0] = -b*ma2[0,1]
    interm2[0,1] = -b*ma2[0,2]
    interm2[1,1] = 1 + b*(-ma2[0,3]-ma2[0,0]*ma2[0,3]+ma2[0,1]*ma2[0,2])
    interm2[0,2] = b*(-ma2[0,4]-ma2[0,3]*ma2[0,4]+ma2[0,2]*ma2[0,5])
    interm2[1,2] = b*(-ma2[0,5]-ma2[0,0]*ma2[0,5]+ma2[0,1]*ma2[0,4])
    interm2[2,0] = 0
    interm2[2,1] = 0
    interm2[2,2] = 1
    # Construct the first term
    ma1 = np.array(warp).reshape([2,3])
    interm1[0,0] = ma1[0,0]
    interm1[0,1] = ma1[0,1]
    interm1[0,2] = ma1[0,2]
    interm1[1,0] = ma1[1,0]
    interm1[1,1] = ma1[1,1]
    interm1[1,2] = ma1[1,2]
    interm1[2,0] = 0
    interm1[2,1] = 0
    interm1[2,2] = 1
    # Getting the results
    temp = np.asmatrix(interm1) * interm2
    pdb.set_trace()
    res = np.array(temp[:2,:]).reshape([1,6])
    return res
# ---------- Efficient Image Sampling and Conversion ---------- #

cdef double bilin_interp(double [:,:] img, double x, double y):
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]

    cdef unsigned int lx = <int>floor(x)
    cdef unsigned int ux = <int>ceil(x)
    cdef unsigned int ly = <int>floor(y)
    cdef unsigned int uy = <int>ceil(y)

    # Need to be a bit careful here due to overflows
    if not (0 <= lx < w and 0 <= ux < w and
            0 <= ly < h and 0 <= uy < h): return 128

    cdef double dx = x - lx
    cdef double dy = y - ly
    return img[ly,lx]*(1-dx)*(1-dy) + \
           img[ly,ux]*dx*(1-dy) + \
           img[uy,lx]*(1-dx)*dy + \
           img[uy,ux]*dx*dy

cpdef double[:] sample_pts_all(double[:,:] img, int resx, int resy, double[:,:] warp, int MModel, double[:,:] tmplt_size = np.empty((0,0))) except *:
    #print MModel
    #print np.array(tmplt_size)
    if MModel == 1:
        return sample_pts(img, resx, resy, warp)
    elif MModel == 5:
        # Using rot with complete motion model
        return sample_pts(img, resx, resy, warp)

cpdef double[:] sample_pts(double[:,:] img, int resx, int resy, double[:,:] warp) except *:
    cdef int n_pts = resx * resy
    cdef double[:] result = np.empty(n_pts, dtype=np.float64)
    cdef int yi, xi, ri
    cdef double y, x, d, wy, wx
    ri = 0
    for yi in range(resy):
        y = <double>yi / (resy-1) - 0.5
        for xi in range(resx):
            x = <double>xi / (resx-1) - 0.5
            d = warp[2,0]*x + warp[2,1]*y + warp[2,2]
            wx = (warp[0,0]*x + warp[0,1]*y + warp[0,2]) / d
            wy = (warp[1,0]*x + warp[1,1]*y + warp[1,2]) / d
            result[ri] = bilin_interp(img, wx, wy)
            ri += 1
    return result

cpdef double [:,:] to_grayscale(unsigned char [:,:,:] img):
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef int d = img.shape[2]
    cdef double [:,:] result = np.empty((h,w), dtype=np.float64)
    cdef int i,j,k
    for i in range(h):
        for j in range(w):
            result[i,j] = 0
            for k in range(d): result[i,j] += img[i,j,k]
            result[i,j] /= d
    return result

# ---------- Numerically estimating sampling jacobian ---------- #

cdef double[:,:] mat_mul(double[:,:] A, double [:,:] B):
    cdef int h = A.shape[0]
    cdef int m = A.shape[1]
    cdef int w = B.shape[1]
    cdef double[:,:] result = np.empty((h,w), dtype=np.float64)
    cdef int i,k,j
    for i in range(h):
        for j in range(w):
            result[i,j] = 0
            for k in range(m):
                result[i,j] += A[i,k]*B[k,j]
    return result

cdef double[:,:] warp_update(double[:,:] old, double [:] update):
    cdef double f,wx,wy,wz
    cdef double[:,:] new_matrix1
    cdef double[:,:] new_matrix2
    cdef double[:,:] res
    f = f_len
    new_matrix1 = np.eye(3,dtype=np.float64)
    new_matrix1[0,2] = update[0]
    new_matrix1[1,2] = update[1]
    new_matrix1[0,0] += update[2]
    new_matrix1[1,1] += update[2]
    
    wx = update[3]
    wy = update[4]
    wz = update[5]
    new_matrix2 = np.eye(3,dtype=np.float64)
    new_matrix2[0,1] = wz
    #new_matrix2[0,2] = f*wy
    new_matrix2[0,2] = wy
    new_matrix2[1,0] = -wz
    #new_matrix2[1,2] = f*wx
    new_matrix2[1,2] = f*wx
    #new_matrix2[2,0] = -wy/f
    new_matrix2[2,0] = -wy
    #new_matrix2[2,1] = -wx/f
    new_matrix2[2,1] = -wx

    res = mat_mul(new_matrix1,old)
    return mat_mul(new_matrix2,res)
'''
cdef double[:] mat_min(double[:] A, double[:] B):
    cdef int h = A.shape[0]:
#    cdef int m = A.shape[1]
#    cdef int w = B.shape[1]
    cdef double[:,:] result = np.empty(h, dtype=np.float64)
    cdef int i
    for i in range(h):
        result[i] = A[i] - B[i]
    return result   
'''

cdef double _eps = 1e-2 #1e-8
cdef double _epsl = 1e-8
cdef double _epsx
cdef double _epsy
cdef double _epsf = 0.8
cdef double img_dir_grad(double[:,:] img, double[:,:] warp, double x, double y, double dx, double dy):
    cdef double wx1, wy1, wx2, wy2, d, ox, oy, offset_size
    offset_size = sqrt(dx*dx + dy*dy)
    
    ox = x + dx
    oy = y + dy
    d = warp[2,0]*ox + warp[2,1]*oy + warp[2,2]
    wx1 = (warp[0,0]*ox + warp[0,1]*oy + warp[0,2]) / d
    wy1 = (warp[1,0]*ox + warp[1,1]*oy + warp[1,2]) / d
    
    ox = x - dx
    oy = y - dy
    d = warp[2,0]*ox + warp[2,1]*oy + warp[2,2]
    wx2 = (warp[0,0]*ox + warp[0,1]*oy + warp[0,2]) / d
    wy2 = (warp[1,0]*ox + warp[1,1]*oy + warp[1,2]) / d
    
    return (bilin_interp(img, wx1, wy1) - bilin_interp(img, wx2, wy2)) / (2 * offset_size)



cpdef double[:,:] sample_pts_grad_batch(double[:,:] img, int resx, int resy, double[:,:] warp, double MModel) except *:
    cdef int xi, yi, i, dims
    cdef double x, y, ox, oy, d, wx, wy, w2x, w2y, Ix, Iy, Ixx, Iyy
    cdef double[:,:] result
    if MModel/100 == 1: dims = 8
    elif MModel/100 == 4: dims = 6
    elif MModel/100 == 6: dims = 2
    elif MModel/100 == 7: dims = 4
    else: print "No MModel match for jacobian"
    result = np.empty((resx*resy, dims), dtype=np.float64)
    i = 0
    for yi in range(resy):
        y = <double>yi / (resy-1) - 0.5
        for xi in xrange(resx):
            x = <double>xi / (resx-1) - 0.5
            #d = warp[2,0]*x + warp[2,1]*y + warp[2,2]
            #wx = (warp[0,0]*x + warp[0,1]*y + warp[0,2]) / d
            #wy = (warp[1,0]*x + warp[1,1]*y + warp[1,2]) / d
            Ix = img_dir_grad(img, warp, x, y, _eps, 0)
            Iy = img_dir_grad(img, warp, x, y, 0, _eps)
            Ixx = Ix * x
            Iyy = Iy * y
            if MModel/100 == 1:
                result[i,0] = Ixx
                result[i,1] = Ix*y
                result[i,2] = Ix
                result[i,3] = Iy*x
                result[i,4] = Iy*y
                result[i,5] = Iy
                result[i,6] = -Ixx*x - Iy*x*y
                result[i,7] = -Ixx*y - Iyy    
            elif MModel/100 == 4:
                result[i,0] = Ixx
                result[i,1] = Ix*y
                result[i,2] = Ix
                result[i,3] = Iy*x
                result[i,4] = Iyy
                result[i,5] = Iy 
            elif MModel/100 == 6:
                result[i,0] = Ix
                result[i,1] = Iy
            elif MModel/100 == 7:
                result[i,0] = Ixx + Iyy
                result[i,1] = -y*Ix + x*Iy
                result[i,2] = Ix
                result[i,3] = Iy
            i += 1
    return result

cpdef double[:,:] sample_pts_grad_homo_ic(double[:,:] img, int resx, int resy, double[:,:] warp) except *:
    cdef int xi, yi, i
    cdef double x, y, ox, oy, d, wx, wy, w2x, w2y, Ix, Iy, Ixx, Iyy
    cdef double[:,:] result = np.empty((resx*resy, 8), dtype=np.float64)
    i = 0
    for yi in range(resy):
        y = <double>yi / (resy-1) - 0.5
        for xi in xrange(resx):
            x = <double>xi / (resx-1) - 0.5
            # Computing the spatial image gradient
            d = warp[2,0]*x + warp[2,1]*y + warp[2,2]
            wx = (warp[0,0]*x + warp[0,1]*y + warp[0,2]) / d
            wy = (warp[1,0]*x + warp[1,1]*y + warp[1,2]) / d
           
            #Ix = img_dir_grad_fwd(img, wx, wy, _epsf, 0)
            #Iy = img_dir_grad_fwd(img, wx, wy, 0, _epsf)
            Ix = img_dir_grad(img, warp, x, y, _eps, 0)
            Iy = img_dir_grad(img, warp, x, y, 0, _eps)
           # Combining image gradient with jacobian of warp function
            Ixx = Ix * x
            Iyy = Iy * y
            result[i,0] = Ixx
            result[i,1] = Ix*y
            result[i,2] = Ix
            result[i,3] = Iy*x
            result[i,4] = Iy*y
            result[i,5] = Iy
            result[i,6] = -Ixx*x - Iy*x*y
            result[i,7] = -Ixx*y - Iyy
            # Next row, please!
            i += 1
    return result


cpdef double[:,:] sample_pts_Jacob(double[:,:] img, int resx, int resy, double[:,:] warp, int MModel, double[:,:] tmplt_size = np.empty((0,0))) except *:
    # These are all for backward methods
    if MModel == 1:
        return sample_pts_grad_sl3(img, resx, resy, warp)
    elif MModel >= 100: # switch for revised DLKT_revise
        return sample_pts_grad_batch(img, resx, resy, warp, MModel)

cdef double f_len = 1.0

cdef double img_dir_grad_fwd(double[:,:] img, double x, double y, double dx, double dy):
    cdef double wx1, wy1, wx2, wy2, d, ox, oy, offset_size
    offset_size = sqrt(dx*dx + dy*dy)
    wx1 = x + dx
    wy1 = y + dy
 
    wx2 = x - dx
    wy2 = y - dy
    return (bilin_interp(img, wx1, wy1) - bilin_interp(img, wx2, wy2)) / (2.0 * offset_size)
    

cpdef double[:,:] sample_pts_grad_sl3(double[:,:] img, int resx, int resy, double[:,:] warp) except *:
    cdef int xi, yi, i
    cdef double x, y, ox, oy, d, w1x, w1y, w2x, w2y, Ix, Iy, Ixx, Iyy
    cdef double[:,:] result = np.empty((resx*resy, 8), dtype=np.float64)
    i = 0
    for yi in range(resy):
        y = <double>yi / (resy-1) - 0.5
        for xi in xrange(resx):
            x = <double>xi / (resx-1) - 0.5
            # Computing the spatial image gradient 
            Ix = img_dir_grad(img, warp, x, y, _epsl, 0)
            Iy = img_dir_grad(img, warp, x, y, 0, _epsl)
            # Combining image gradient with jacobian of warp function
            Ixx = Ix * x
            Iyy = Iy * y
            result[i,0] = Ixx - Iyy
            result[i,1] = Ix*y
            result[i,2] = Ix
            result[i,3] = Iy*x
            result[i,4] = Ixx + 2*Iyy
            result[i,5] = Iy
            result[i,6] = -Ixx*x - Iyy*x
            result[i,7] = -Ixx*y - Iyy*y
            # Next row, please!
            i += 1
    return result

cdef double[:] scv_intensity_map(double[:] src, double[:] dst):
    cdef int n_pts = src.shape[0]
    cdef double[:,:] P = np.zeros((256,256), dtype=np.float64)
    cdef int k, i, j
    for k in range(n_pts):
        i = <int>src[k]
        j = <int>dst[k]
        P[i,j] += 1
    
    cdef double[:] intensity_map = np.zeros(256, dtype=np.float64)
    cdef double normalizer, total
    for i in range(256):
        normalizer = 0
        total = 0
        for j in range(256):
            total += j * P[i,j]
            normalizer += P[i,j]
        if normalizer > 0: intensity_map[i] = total / normalizer
        else: intensity_map[i] = i
    return intensity_map

cdef double[:] scv_expected_img(double[:] img, double[:] intensity_map):
    cdef int n_pts = img.shape[0]
    cdef double[:] result = np.empty(n_pts, dtype=np.float64)
    cdef int i
    for i in range(n_pts):
        result[i] = intensity_map[<int>img[i]]
    return result

cdef normalize_hom(double[:,:] H):
    cdef int i, j
    for i in range(3):
        for j in range(3):
            H[i,j] = H[i,j] / H[2,2]
    

cpdef double[:,:] compute_homography(double[:,:] in_pts, double[:,:] out_pts):
    cdef int num_pts = in_pts.shape[1]
    cdef double[:,:] constraint_matrix = np.empty((num_pts*2,9), dtype=np.float64)
    cdef int i, r1, r2
    for i in range(num_pts):
        r1 = 2*i
        constraint_matrix[r1,0] = 0
        constraint_matrix[r1,1] = 0
        constraint_matrix[r1,2] = 0
        constraint_matrix[r1,3] = -in_pts[0,i]
        constraint_matrix[r1,4] = -in_pts[1,i]
        constraint_matrix[r1,5] = -1
        constraint_matrix[r1,6] = out_pts[1,i] * in_pts[0,i]
        constraint_matrix[r1,7] = out_pts[1,i] * in_pts[1,i]
        constraint_matrix[r1,8] = out_pts[1,i]

        r2 = 2*i + 1
        constraint_matrix[r2,0] = in_pts[0,i]
        constraint_matrix[r2,1] = in_pts[1,i]
        constraint_matrix[r2,2] = 1
        constraint_matrix[r2,3] = 0
        constraint_matrix[r2,4] = 0
        constraint_matrix[r2,5] = 0
        constraint_matrix[r2,6] = -out_pts[0,i] * in_pts[0,i]
        constraint_matrix[r2,7] = -out_pts[0,i] * in_pts[1,i]
        constraint_matrix[r2,8] = -out_pts[0,i]
    U,S,V = np.linalg.svd(constraint_matrix)
    cdef double[:,:] H = V[8].reshape(3,3) / V[8][8]
    return H

cpdef double[:,:] compute_affine(double[:,:] in_pts, double[:,:] tmplt_size, double[:,:] out_pts):
    cdef int num_pts = in_pts.shape[1]
    cdef double[:,:] constraint_matrix = np.empty((num_pts*2,6),dtype=np.float64)
    cdef double[:,:] res_matrix = np.empty((num_pts*2,1),dtype=np.float64)
    cdef int i, r1, r2

    for i in range(num_pts):
        r1 = 2*i
        constraint_matrix[r1,0] = in_pts[0,i]
        constraint_matrix[r1,1] = in_pts[1,i]
        constraint_matrix[r1,2] = 1
        constraint_matrix[r1,3] = 0
        constraint_matrix[r1,4] = 0
        constraint_matrix[r1,5] = 0
        res_matrix[r1,0] = out_pts[0,i]

        r2 = 2*i + 1
        constraint_matrix[r2,0] = 0
        constraint_matrix[r2,1] = 0
        constraint_matrix[r2,2] = 0       
        constraint_matrix[r2,3] = in_pts[0,i]
        constraint_matrix[r2,4] = in_pts[1,i]
        constraint_matrix[r2,5] = 1      
        res_matrix[r2,0] = out_pts[1,i]
    X = np.linalg.lstsq(constraint_matrix,res_matrix)[0]
    cdef double[:,:] H = X.reshape(1,6)
    return H


def rectangle_to_region(ul, lr):
    return np.array([ul, [lr[0], ul[1]], lr, [ul[0], lr[1]]],
                    dtype = np.float64).T

# ---------- Legacy Codes From Previous Implementation ---------- #
# TODO: Replace these with optimized versions. The only operations
#       we need are compute_homography, square_to_corners_warp, and
#       some way to get the image of the centered unit square under
#       a homography (this is the only current use of apply_to_pts)x

def homogenize(pts):
    (h,w) = pts.shape
    results = np.empty((h+1,w))
    results[:h] = pts
    results[-1].fill(1)
    return results

def dehomogenize(pts):
    (h,w) = pts.shape
    results = np.empty((h-1,w))
    results[:h-1] = pts[:h-1]/pts[h-1]
    return results

# def compute_homography(in_pts, out_pts):
#     num_pts = in_pts.shape[1]
#     in_pts = homogenize(in_pts)
#     out_pts = homogenize(out_pts)
#     constraint_matrix = np.empty((num_pts*2, 9))
#     for i in xrange(num_pts):
#         p = in_pts[:,i]
#         q = out_pts[:,i]
#         constraint_matrix[2*i,:] = np.concatenate([[0,0,0], -p, q[1]*p], axis=1)
#         constraint_matrix[2*i+1,:] = np.concatenate([p, [0,0,0], -q[0]*p], axis=1)
#     U,S,V = np.linalg.svd(constraint_matrix)
#     homography = V[8].reshape((3,3))
#     homography /= homography[2,2]
#     return np.asmatrix(homography)

_square = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T
def square_to_corners_warp(corners, MModel=1):
    if MModel == 1:
        return compute_homography(_square, corners)
    elif MModel == 4:
        return compute_affine(_square, np.zeros((1,6)), corners) 

def apply_to_pts_all(warp_matrix, pts, MModel, tmplt_size=np.empty((0,0))):
    if MModel == 1:
        return apply_to_pts(warp_matrix, pts)

def apply_to_pts(homography, pts):
    
    (h,w) = pts.shape    
    result = np.empty((h+1,w))
    result[:h] = pts
    result[h].fill(1)
    result = np.asmatrix(homography) * result
    result[:h] = result[:h] / result[h]
    result[np.isnan(result)] = 0
    return np.asarray(result[:h])

'''
def apply_to_pts_aff(warp_matrix, pts, tmplt_size):
    result = np.empty((3,4))
    result[-1].fill(1)
    result[0,0] = 1
    result[1,0] = 1
    result[0,1] = tmplt_size[0,2] + 1
    result[1,1] = 1
    result[0,2] = tmplt_size[0,2] + 1
    result[1,2] = tmplt_size[0,3] + 1
    result[0,3] = 1
    result[1,3] = tmplt_size[0,3] + 1

    res = np.asmatrix(warp_matrix.reshape((2,3))) * result
    return np.asarray(res)
 
    (h,w) = pts.shape    
    result = np.empty((h+1,w))
    result[:h] = pts
    result[-1].fill(1)
    res = np.asmatrix(warp_matrix.reshape((2,3))) * result
    res[np.isnan(res)] = 0
    return np.asarray(res)

'''
def apply_to_pts_aff(warp_matrix, pts):
    (h,w) = pts.shape    
    result = np.empty((h+1,w))
    result[:h] = pts
    result[-1].fill(1)
#    print warp_matrix.shape
    res = np.asmatrix(warp_matrix.reshape((2,3))) * result
    res[np.isnan(res)] = 0
    return np.asarray(res)


def draw_region(img, corners, color, thickness=1, draw_x=False):
    for i in xrange(4):
        p1 = (int(corners[0,i]), int(corners[1,i]))
        p2 = (int(corners[0,(i+1)%4]), int(corners[1,(i+1)%4]))
        cv2.line(img, p1, p2, color, thickness)
    if draw_x:
        for i in xrange(4):
            p1 = (int(corners[0,i]), int(corners[1,i]))
            p2 = (int(corners[0,(i+2)%4]), int(corners[1,(i+2)%4]))
            cv2.line(img, p1, p2, color, thickness)

def polygon_descriptors(corners):
    """ Computes the area, perimeter, and center of mass of a polygon.
    
    Parameters:
    -----------
    corners : (2,n) numpy array
      The vertices of the polygon. Should be in clockwise or
      counter-clockwise order.
    
    Returns:
    --------
    A tuple (perimeter, area, (center of mass x, center of mass y)).
    """
    n_points = corners.shape[1]
    p, a, cx, cy = 0, 0, 0, 0
    for i in xrange(n_points):
        j = (i+1) % n_points
        dot = corners[0,i]*corners[1,j] - corners[0,j]*corners[1,i]
        a += dot
        cx += (corners[0,i] + corners[0,j]) * dot
        cy += (corners[1,i] + corners[1,j]) * dot
        p += np.linalg.norm(corners[:,i] - corners[:,j])
    a /= 2
    cx /= 6*a
    cy /= 6*a
    a = abs(a)
    return (p, a, (cx,cy))

