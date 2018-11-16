""" 
Implementation of the Baker+Matthews Inverse Compositional Tracker.
Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import cv2

cimport numpy as np
import numpy as np
from scipy.linalg import pinv
from utility import apply_to_pts_all, square_to_corners_warp
from utility cimport *
import pdb

cdef class DLKTracker:

    cdef:
        int max_iters
        double threshold, ssigma
        int resx
        int resy
        np.ndarray template, H_inv, J
        double[:,:] current_warp
        double[:] intensity_map
        bint use_scv
        bint initialized
        double[:,:] init_img
        double[:,:] init_x_pos
        double[:,:] init_y_pos
        double[:,:] tmplt_size
        double[:,:] x_pos
        double[:,:] y_pos
        double[:,:] mask
        double[:] ncc
        int MModel
        int Algo
        dict options      

    def __init__(self, int max_iters, double threshold, int resx, int resy, bint use_scv,
                 int MModel = 1, int Algo = 1):
        self.max_iters = max_iters
        self.threshold = threshold
        self.resx = resx
        self.resy = resy
        self.use_scv = use_scv
        self.initialized = False
        self.MModel = MModel
        self.Algo = Algo
        self.current_warp = np.empty((0,0))
        self.tmplt_size = np.empty((0,0))
        self.x_pos = np.empty((0,0))
        self.y_pos = np.empty((0,0))
        self.mask = np.empty((0,0))
        self.ncc = np.ones((resy*resx),dtype=np.float64)
        # Jesse
        self.options = {(1,1):'bhomo', # 8 DOF
                        (4,1):'baffn', # 6 DOF
                        (5,1):'bsalt', # 3 DOF
                        (6,1):'btran', # 2 DOF
                        (7,1):'bsimt'} # 4 DOF

        print 'Initializing DLKTracker tracker with:'
        print 'max_iters: ', self.max_iters
        print 'threshold: ', self.threshold
        print 'resx: ', self.resx
        print 'resy: ', self.resy
        print 'use_scv: ', self.use_scv
        print 'MModel: ', self.MModel
        print 'Algo: ', self.Algo

    cpdef initialize(self, double[:,:] img, double[:,:] region_corners):
        self.initialized = False

        opt = self.options[(self.MModel,self.Algo)]
        print opt
        self.current_warp = square_to_corners_warp(np.asarray(region_corners),1)        
        self.template = np.asarray(sample_pts_all(img, self.resx, self.resy, self.current_warp, 1))
        if opt[0] == 'b':
            self.J = np.asmatrix(sample_pts_Jacob(img, self.resx, self.resy, self.current_warp, self.MModel*100))
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
        cdef double[:,:] Jpc, R_error, spatio_error
        cdef double[:] sampled_img
        cdef double[:] xy_tmp
        cdef long[:] indx_sort
        cdef double x_mean, y_mean
        opt = self.options[(self.MModel,self.Algo)]
        for i in range(self.max_iters):
            sampled_img = sample_pts_all(img, self.resx, self.resy, self.current_warp, 1)
            try:
                self.H_inv = (self.J.T * self.J).I
            except:
                self.H_inv = np.zeros((self.J.shape[1],self.J.shape[1]),dtype=np.float64)
                self.H_inv = np.asmatrix(self.H_inv)
            if self.use_scv:
                if self.intensity_map == None: self.intensity_map = scv_intensity_map(sampled_img, self.template)
                sampled_img = scv_expected_img(sampled_img, self.intensity_map)

            error = np.asarray(sampled_img-self.template).reshape(-1,1)

            update = self.J.T * error
            update = self.H_inv * update
            update = np.asarray(update).squeeze()

            temp_matrix = np.zeros((1,9),dtype=np.float64)
            if self.MModel == 1:
                temp_matrix[0,:8] = update[:]
                temp_matrix = temp_matrix.reshape(3,3)
                temp_matrix[0,0] += 1
                temp_matrix[1,1] += 1
                temp_matrix[2,2] += 1
            elif self.MModel == 4:
                temp_matrix[0,:6] = update[:]
                temp_matrix = temp_matrix.reshape(3,3)
                temp_matrix[0,0] += 1
                temp_matrix[1,1] += 1
                temp_matrix[2,2] += 1
            elif self.MModel == 6:
                temp_matrix = temp_matrix.reshape(3,3)
                temp_matrix[0,2] = update[0]
                temp_matrix[1,2] = update[1]
                temp_matrix[0,0] += 1
                temp_matrix[1,1] += 1
                temp_matrix[2,2] += 1
            elif self.MModel == 7:
                scale = update[0] + 1.0
                thd = update[1]
                tx = update[2]
                ty = update[3]
                temp_matrix = temp_matrix.reshape(3,3)
                temp_matrix[0,0] =  scale * np.cos(thd)
                temp_matrix[0,1] = -scale * np.sin(thd)
                temp_matrix[1,0] =  scale * np.sin(thd)
                temp_matrix[1,1] =  scale * np.cos(thd)
                temp_matrix[0,2] = tx
                temp_matrix[1,2] = ty
                temp_matrix[2,2] = 1.0                
            self.current_warp = np.asmatrix(self.current_warp) * pinv(temp_matrix)
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
        if self.MModel < 5:
            self.current_warp = warp
        
        if reset_intensity: self.intensity_map = None

    cpdef double[:,:] get_warp(self):
        return np.asmatrix(self.current_warp)

    cpdef double[:,:] get_tmplt(self):
        return np.asarray(self.template.reshape(self.resx,self.resy))

    cpdef double[:,:] get_jacob(self):
        return np.asarray(self.J)
    
    cpdef double[:,:] get_size(self):
        return np.asarray(self.tmplt_size)

    cpdef set_region(self, double[:,:] corners, bint reset_intensity=True):
        self.current_warp = square_to_corners_warp(corners, self.MModel)
        if reset_intensity: self.intensity_map = None

    cpdef get_region(self):
        return apply_to_pts_all(np.asarray(self.get_warp()), np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T, 1)

