"""
Implementation of the Nearest Neighbour Tracking Algorithm.
Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import os
import shelve
import threading
from build_graph import build_graph
from search_graph import search_graph 

cimport numpy as np
import numpy as np

import pyflann
# For building build_graph
#from knnsearch import knnsearch
#import random
#import numpy
#from numpy import matrix as MA

from utility import apply_to_pts, apply_to_pts_all, square_to_corners_warp
from utility cimport *

#_stored_warps_lock = threading.Lock()
#_stored_warps = shelve.open(os.path.expanduser("~/warp_cache_dpp"))
#_stored_warps = shelve.open(os.path.expanduser("~/warp_cache"))
cdef class _WarpIndex_Flann:
    cdef:
        double[:,:,:] warps
        double[:,:] images
        object flann
        bint verbose
        double[:,:] nodes
        bint gnn

    def __init__(self, double[:,:] img, double[:,:] warp, 
                 int n_samples, int resx, int resy, 
                 double sigma_t, double sigma_d, int MModel = 1, bint gnn = False,
                 double[:,:] x_pos = np.empty((0,0)), double[:,:] y_pos = np.empty((0,0)), bint verbose = False):
        self.verbose = verbose
        self.nodes = None
        self.gnn = gnn
        pyflann.set_distance_type("manhattan")

        # --- Sampling Warps --- #
        self._msg("Sampling Warps...")
        warp_key = "%d %d %.5g %.5g" % (n_samples, MModel, sigma_t, sigma_d)
        self._msg("Warp Key = %s" % warp_key)
        #if not _stored_warps.has_key(warp_key):
            #with _stored_warps_lock:
        if MModel == 1:
            warps = np.empty((n_samples,3,3), dtype=np.float64)
            for i in range(n_samples):
                warps[i,:,:] = _random_homography(sigma_t, sigma_d)
        elif MModel == 4:
            warps = np.empty((n_samples,3,3), dtype=np.float64)
            for i in range(n_samples):
                warps[i,:,:] = _random_affine(sigma_t, sigma_d)
        elif MModel >= 5:
            warps = np.empty((n_samples,3,3), dtype=np.float64)
            for i in range(n_samples):
                warps[i,:,:] = _random_ldof(sigma_t, sigma_d, MModel)
                #_stored_warps[warp_key] = warps
        #self.warps = _stored_warps[warp_key]
        self.warps = warps
         # --- Sampling Images --- #
        self._msg("Sampling Images...")
        cdef int n_pts = resx * resy
        self.images = np.empty((n_pts, n_samples), dtype=np.float64)
        print n_samples
        for i in range(n_samples):
            inv_warp = np.asmatrix(self.warps[i,:,:]).I
            self.images[:,i] = sample_pts_all(img, resx, resy, mat_mul(warp, inv_warp),1)
        if not self.gnn:
        # --- Building Flann Index --- #
            self._msg("Building Flann Index...")
            self.flann = pyflann.FLANN()
        #self.flann.build_index(np.asarray(self.images).T, algorithm='linear')
            self.flann.build_index(np.asarray(self.images).T, algorithm='kdtree', trees=6, checks=50)
            self._msg("Done!")
        elif self.gnn:
            self.images = np.asmatrix(self.images,'f8')
            self.nodes = build_graph(self.images.T,400)

    cpdef _msg(self, str):
        if self.verbose: print str

    cpdef best_match(self, img):
        if self.gnn:
            nn_id,b,c = search_graph(img,self.nodes,self.images.T,1)
            return self.warps[<int>(nn_id),:,:]
        else:
            results, dists = self.flann.nn_index(np.asarray(img))
            return self.warps[<int>results[0],:,:]

    cpdef mean_pixel_variance(self):
        return np.mean(np.var(self.images, 1))

cdef class DNNTracker:

    cdef:
        _WarpIndex_Flann warp_index
        int max_iters
        int resx, resy
        np.ndarray template
        int n_samples
        int MModel
        double sigma_t, sigma_d
        double[:,:] current_warp
        double[:,:] current_affine
        double[:] intensity_map
        bint use_scv
        bint verbose
        bint initialized
        bint gnn
        bint exp
        double[:,:] x_pos
        double[:,:] y_pos

    def __init__(self, int max_iters, int n_samples, int resx, int resy, double sigma_t, 
                 double sigma_d, bint use_scv, int MModel = 1, bint gnn = False, bint exp = False, bint verbose=True):
        self.max_iters = max_iters
        self.n_samples = n_samples
        self.resx = resx
        self.resy = resy
        self.sigma_t = sigma_t
        self.sigma_d = sigma_d
        self.use_scv = use_scv
        self.verbose = verbose
        self.initialized = False
        self.MModel = MModel
        self.gnn = gnn
        self.exp = exp
        #self.template = np.empty((0))
        self.x_pos = np.empty((0,0))
        self.y_pos = np.empty((0,0))
        print "Initializing DNN tracker with:"
        print " max_iters=", max_iters
        print " n_samples=", n_samples
        print " res=({:d}, {:d})".format(resx, resy)
        print 'use_scv=', use_scv
        print 'sigma_t=', sigma_t
        print 'sigma_d=', sigma_d
        print 'MModel=', MModel
        print 'gnn=', gnn
        print 'exp=', exp

    cpdef initialize(self, double[:,:] img, double[:,:] region_corners):
        self.initialized = False
        self.current_warp = square_to_corners_warp(np.asarray(region_corners), 1)
        self.template = np.asarray(sample_pts_all(img, self.resx, self.resy, self.current_warp, 1))
        self.warp_index = _WarpIndex_Flann(img, self.current_warp,
                                           self.n_samples, self.resx, self.resy,
                                           self.sigma_t, self.sigma_d, self.MModel, self.gnn, self.x_pos, self.y_pos, verbose=self.verbose)
        if self.use_scv:
            self.intensity_map = np.arange(256, dtype=np.float64)
        self.initialized = True

    cpdef initialize_with_rectangle(self, double[:,:] img, ul, lr):
        cpdef double[:,:] region_corners = \
            np.array([[ul[0], ul[1]],
                      [lr[0], ul[1]],
                      [lr[0], lr[1]],
                      [ul[0], lr[1]]], dtype=np.float64).T
        self.initialize(img, region_corners)

    cpdef update(self, double[:,:] img):
        if not self.initialized: return
        cdef int i
        cdef double[:] sampled_img
        for i in range(self.max_iters):
            sampled_img = sample_pts_all(img, self.resx, self.resy, self.current_warp, 1)
            if self.use_scv:
                if self.intensity_map == None: self.intensity_map = scv_intensity_map(sampled_img, self.template)
                sampled_img = scv_expected_img(sampled_img, self.intensity_map)
            update = self.warp_index.best_match(sampled_img)
            self.current_warp = mat_mul(self.current_warp, update)
            normalize_hom(self.current_warp)
        if self.use_scv:
            sampled_img = sample_pts_all(img, self.resx, self.resy, self.current_warp, 1)
            self.intensity_map = scv_intensity_map(sampled_img, self.template)

    cpdef set_intensity_map(self, double[:] intensity_map):
        self.intensity_map = intensity_map

    cpdef double[:] get_intensity_map(self):
        return self.intensity_map

    cpdef is_initialized(self):
        return self.initialized

    cpdef set_warp(self, double[:,:] warp, bint reset_intensity=True):
        self.current_warp = warp
        if reset_intensity: self.intensity_map = None

    cpdef double[:,:] get_warp(self):
        return np.asmatrix(self.current_warp)

    cpdef set_region(self, double[:,:] corners, bint reset_intensity=True):
        self.current_warp = square_to_corners_warp(corners, 1)
        if reset_intensity: self.intensity_map = None

    cpdef get_region(self):
        return apply_to_pts_all(np.asarray(self.get_warp()), np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T, 1)

    cpdef get_warp_index(self):
        return self.warp_index

cdef double[:,:] _square = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T
cdef double[:,:] _random_homography(double sigma_t, double sigma_d):
    cdef double[:,:] disturbed = np.random.normal(0,sigma_d, (2,4)) + np.random.normal(0, sigma_t, (2,1)) + _square
    cdef double[:,:] H = compute_homography(_square, disturbed)
    return H

cdef double[:,:] _square_aff = np.array([[-.5,-.5],[.5,-.5],[-.5,.5]]).T
cdef double[:,:] _random_affine(double sigma_t, double sigma_d):
    cdef double[:,:] disturbed = np.random.normal(0, sigma_d, (2,3)) + np.random.normal(0, sigma_t, (2,1)) + _square_aff
    cdef double[:,:] H = compute_affine(_square_aff, np.zeros((1,6)), disturbed)
    cdef double[:,:] Aff_H = np.zeros((3,3), dtype=np.float64)
    Aff_H[0,:] = H[0,:3]
    Aff_H[1,:] = H[0,3:]
    Aff_H[2,2] = 1.0
    return Aff_H

cdef double[:,:] _random_ldof_inconsistant(double sigma_t, double sigma_d, int MModel):
    cdef double[:,:] H = np.eye(3,dtype=np.float64)

    if MModel == 6:
        H[0,-1] = np.random.normal(0, sigma_t, (1))[0]
        H[1,-1] = np.random.normal(0, sigma_t, (1))[0]
    elif MModel == 5:
        H[0,-1] = np.random.normal(0, sigma_t, (1))[0]
        H[1,-1] = np.random.normal(0, sigma_t, (1))[0]
        H[0, 0] += np.random.normal(0, sigma_d, (1))[0] 
        H[1, 1] = H[0, 0]
    elif MModel == 7:
        thd = np.random.normal(0, sigma_d, (1))[0]
        scale = 1.0 + np.random.normal(0, sigma_d, (1))[0]
        tx = np.random.normal(0, 10*sigma_t, (1))[0]
        ty = np.random.normal(0, 10*sigma_t, (1))[0]
        H[0,0] = scale * np.cos(thd)
        H[0,1] = -scale * np.sin(thd)
        H[0,2] = tx
        H[1,0] = scale * np.sin(thd)
        H[1,1] = scale * np.cos(thd)
        H[1,2] = ty
    return H

def getDecomposedAffineMatrices(tx, ty, sin_theta, cos_theta, s, a, b):
    trans_mat = np.mat(
        [[1, 0, tx],
         [0, 1, ty],
         [0, 0, 1]]
    )
    rot_mat = np.mat(
        [[cos_theta, - sin_theta, 0],
         [sin_theta, cos_theta, 0],
         [0, 0, 1]]
    )
    scale_mat = np.mat(
        [[1 + s, 0, 0],
         [0, 1 + s, 0],
         [0, 0, 1]]
    )
    shear_mat = np.mat(
        [[1 + a, b, 0],
         [0, 1, 0],
         [0, 0, 1]]
    )
    return (trans_mat, rot_mat, scale_mat, shear_mat)

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

    (trans_mat, rot_mat, scale_mat, shear_mat) = getDecomposedAffineMatrices(tx, ty, sin_theta, cos_theta, s, a, b)

    rt_mat = rot_mat * trans_mat
    se2_mat = scale_mat * rot_mat * trans_mat
    affine_rec_mat = shear_mat * scale_mat * rot_mat * trans_mat

    return (trans_mat, rt_mat, se2_mat, affine_rec_mat)

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
    b2 = np.mat(U.transpose()) * np.mat(b)
    y = np.zeros(b2.shape, dtype=np.float64)
    for j in xrange(b2.shape[0]):
        y[j] = b2[j] / S[j]
    x = np.mat(V.transpose()) * np.mat(y)
    x_mat = x.reshape(2, 3)
    affine_mat = np.zeros((3, 3), dtype=np.float64)

    affine_mat[0, :] = x_mat[0, :]
    affine_mat[1, :] = x_mat[1, :]
    affine_mat[2, 2] = 1.0
    return np.mat(affine_mat)

cdef double[:,:] _random_ldof(double sigma_t, double sigma_d, int MModel):
    cdef double[:,:] H = np.eye(3,dtype=np.float64)
    cdef double[:,:] disturbed = np.random.normal(0,sigma_d, (2,4)) + np.random.normal(0, sigma_t, (2,1)) + _square
    if MModel == 6:
        dist_x = np.mean(disturbed[0, :])
        dist_y = np.mean(disturbed[1, :])        
        H[0,-1] = dist_x
        H[1,-1] = dist_y
    elif MModel == 7:
        # should normalize  first
        dist_x = np.mean(disturbed[0, :])
        dist_y = np.mean(disturbed[1, :]) 
        dist_temp = disturbed.copy()
        dist_temp[0, :] -= dist_x
        dist_temp[1, :] -= dist_y
        affine_mat_ls = computeAffineLS(_square, dist_temp)
        (trans_mat, rt_mat, se2_mat, affine_rec_mat) = decomposeAffineInverse(affine_mat_ls)
        trans_mat_inv = np.mat(
            [[1, 0, dist_x],
             [0, 1, dist_y],
             [0, 0, 1]]
        )
        H = trans_mat_inv * se2_mat
    return H


'''
cdef double[:,:] build_graph(np.ndarray[np.float, ndim=2] X, np.int k):

   dt = numpy.dtype('f8')
    f=[]
    nodes = numpy.zeros((X.shape[0],k),dt)
    
    for i in range(X.shape[0]):
        query = MA(X[(i-1),0:],dt)
        (nns_inds, nns_dists) = knnsearch(query,X,k+1)
        I  = 0 
        f = []
        while I < len(nns_inds):
            if nns_inds[I] == i-1:
                nns_inds.remove(i-1)
                nodes[i-1,0:] = nns_inds
                break            
            else:
                I += 1
    return nodes 
'''
        
