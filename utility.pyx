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
    return expm(A)

# ---------- Efficient Image Sampling and Conversion ---------- #

cpdef double bilin_interp(double [:,:] img, double x, double y):
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

cdef double _eps = 1e-8
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
            Ix = img_dir_grad(img, warp, x, y, _eps, 0)
            Iy = img_dir_grad(img, warp, x, y, 0, _eps)
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
    results[h].fill(1)
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
def square_to_corners_warp(corners):
    return compute_homography(_square, corners)

def apply_to_pts(homography, pts):
    (h,w) = pts.shape    
    result = np.empty((h+1,w))
    result[:h] = pts
    result[h].fill(1)
    result = np.asmatrix(homography) * result
    result[:h] = result[:h] / result[h]
    result[np.isnan(result)] = 0
    return np.asarray(result[:h])

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
