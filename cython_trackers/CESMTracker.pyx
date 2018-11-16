""" 
Implementation of the ESM Tracker

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import cv2

cimport numpy as np
import numpy as np
import pdb

from scipy.linalg import pinv
from nntracker.utility import apply_to_pts_all, square_to_corners_warp
from nntracker.utility cimport *

cdef class CESMTracker:

    cdef:
        int max_iters
        double threshold
        int resx
        int resy
        np.ndarray template, Je
        double[:,:] current_warp
        double[:] intensity_map
        bint use_scv
        bint initialized
        int MModel
        

    def __init__(self, int max_iters, double threshold, int resx, int resy, bint use_scv, int MModel):
        self.max_iters = max_iters
        self.threshold = threshold
        self.resx = resx
        self.resy = resy
        self.use_scv = use_scv
        self.initialized = False
        self.MModel = MModel

    cpdef initialize(self, double[:,:] img, double[:,:] region_corners):
        self.initialized = False
        self.current_warp = square_to_corners_warp(np.asarray(region_corners), 1)
        self.template = np.asarray(sample_pts_all(img, self.resx, self.resy, self.current_warp, 1))
        
        self.Je = _estimate_jacobian_corner(img, self.MModel, self.resx, self.resy, self.current_warp)
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
        cdef double[:,:] Jpc
        cdef double[:] sampled_img
        for i in range(self.max_iters):
            sampled_img = sample_pts_all(img, self.resx, self.resy, self.current_warp, 1)
            if self.use_scv:
                if self.intensity_map == None: self.intensity_map = scv_intensity_map(sampled_img, self.template)
                sampled_img = scv_expected_img(sampled_img, self.intensity_map)
            error = np.asarray(sampled_img - self.template).reshape(-1,1)
            Jpc = _estimate_jacobian_corner(img, self.MModel, self.resx, self.resy, self.current_warp)
            for item in range(8):
                temp = np.array(Jpc[:,item])
                #temp = np.absolute(temp.reshape((100,100)))
                temp = temp.reshape((100,100))
                cv2.imwrite(str(item)+'.jpg', temp)
            J = np.asmatrix(Jpc + self.Je) / 2.0
            update = np.asarray(np.linalg.lstsq(J, error)[0]).squeeze()
            new_corners = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T
            for indx in xrange(8):
                new_corners[indx//4, indx%4] -= update[indx]
            H_update = square_to_corners_warp(new_corners, 1)
            self.current_warp = self.current_warp * np.asmatrix(H_update)
            normalize_hom(self.current_warp)            
            if np.sum(np.abs(update)) < self.threshold: break
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
        self.current_warp = square_to_corners_warp(corners, self.MModel)
        if reset_intensity: self.intensity_map = None

    cpdef set_tmplt_exp(self, double[:,:] img, double[:,:] regions):
        self.template = np.asarray(sample_pts_all(img, self.resx, self.resy, square_to_corners_warp(np.asarray(regions)), 1))
        self.Je = np.asmatrix(sample_pts_Jacob(img, self.resx, self.resy, square_to_corners_warp(np.asarray(regions), self.MModel), 10*self.MModel+1))    

    cpdef get_region(self):
        return apply_to_pts_all(np.asarray(self.get_warp()), np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T, 1)
        

def _estimate_jacobian_corner(img, MModel, resx, resy, initial_warp, eps=1e-8):
    def f(p):
        W = initial_warp * np.asmatrix(p)
        K = sample_pts_all(img, resx, resy, W, 1)
        return np.asarray(K)
    def est_jacob(corners):
        H = square_to_corners_warp(corners, 1)  # 2*4 matrix
        return f(H)
    original_corners = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T
    jacobian = np.empty((resx*resy, 8))
    for indx in xrange(8):
        original_corners[indx//4, indx%4] += eps
        patch1 = est_jacob(original_corners)
        original_corners[indx//4, indx%4] -= 2*eps
        patch2 = est_jacob(original_corners)
        original_corners[indx//4, indx%4] += eps
        jacobian[:,indx] = (patch1 - patch2) / (2*eps)
    return np.asmatrix(jacobian)

