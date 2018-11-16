""" 
Implementation of the ESM Tracker

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import cv2

cimport numpy as np
import numpy as np

from utility import apply_to_pts, square_to_corners_warp
from utility cimport *

cdef class ESMTracker:

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
        

    def __init__(self, int max_iters, double threshold, int resx, int resy, bint use_scv,
                 int write_log = 0, char* multi_approach = "none"):
        self.max_iters = max_iters
        self.threshold = threshold
        self.resx = resx
        self.resy = resy
        self.use_scv = use_scv
        self.initialized = False

        print 'Initializing Cython ESM tracker with:'
        print 'max_iters: ', self.max_iters
        print 'threshold: ', self.threshold
        print 'resx: ', self.resx
        print 'resy: ', self.resy
        print 'use_scv: ', self.use_scv

    cpdef initialize(self, double[:,:] img, double[:,:] region_corners):
        self.initialized = False
        self.current_warp = square_to_corners_warp(np.asarray(region_corners))
        self.template = np.asarray(sample_pts(img, self.resx, self.resy, self.current_warp))
        self.Je = np.asmatrix(sample_pts_grad_sl3(img, self.resx, self.resy, self.current_warp))
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
            sampled_img = sample_pts(img, self.resx, self.resy, self.current_warp)
            if self.use_scv:
                if self.intensity_map == None: self.intensity_map = scv_intensity_map(sampled_img, self.template)
                sampled_img = scv_expected_img(sampled_img, self.intensity_map)
            error = np.asarray(self.template - sampled_img).reshape(-1,1)
            Jpc = sample_pts_grad_sl3(img, self.resx, self.resy, self.current_warp)
            J = np.asmatrix(Jpc + self.Je) / 2.0
            update = np.asarray(np.linalg.lstsq(J, error)[0]).squeeze()
            self.current_warp = mat_mul(self.current_warp, make_hom_sl3(update))
            normalize_hom(self.current_warp)
            if np.sum(np.abs(update)) < self.threshold: break
        if self.use_scv:
            sampled_img = sample_pts(img, self.resx, self.resy, self.current_warp)
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
        self.current_warp = square_to_corners_warp(corners)
        if reset_intensity: self.intensity_map = None

    cpdef get_region(self):
        return apply_to_pts(self.get_warp(), np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T)
        


