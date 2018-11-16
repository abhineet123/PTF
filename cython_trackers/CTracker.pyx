import cv2
import pdb
import numpy as np
import math

from scipy.linalg import pinv
from utility import apply_to_pts_all, square_to_corners_warp
from utility cimport *

class CTracker:
    def __init__(self, max_iters, threshold, resx, resy, use_scv, MModel=1, Algo=1):
        self.max_iters = max_iters
        self.threshold = threshold
        self.MModel = MModel
        self.Algo = Algo
        self.resx = resx
        self.resy = resy
        self.use_scv = use_scv
        self.intensity_map = None
        self.initialized = False

        self.pos = None
        self.current_warp = None
        self.error = None 
        self.prev_img = None
        self.p0 = None

        self.origin_img = None
        self.lk_params = dict(winSize = (10,10),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.use_ransac = True
        print "Initializing RKLT tracker with:"
        print " max_iters=", max_iters
        print " threshold=", threshold
        print " res=({:d}, {:d})".format(resx, resy)
        print 'use_scv=', use_scv
        print 'Algo=', Algo
        print 'MModel=', MModel

    def initialize(self, img, region_corners):
        self.initialized = False
        self.prev_img = img.astype(np.uint8)
        self.origin_img = img.astype(np.uint8)
        self.current_warp = square_to_corners_warp(np.asarray(region_corners))
        temp_x = np.linspace(-.5,.5,self.resx)
        temp_y = np.linspace(-.5,.5,self.resy)
        xv, yv = np.meshgrid(temp_x, temp_y)
        self.pos = np.empty((2,self.resx*self.resy),dtype=np.float32)
        self.pos[0,:] = xv.flatten()[:]
        self.pos[1,:] = yv.flatten()[:]
        self.p0 = apply_to_pts_all(self.current_warp, self.pos, 1)
        self.p0 = self.p0.T.reshape(self.resx*self.resy,1,2)
        self.p0 = self.p0.astype(np.float32)        
        self.pval = np.copy(self.p0)
        # Initialization for IC tracker
        self.template = np.asarray(sample_pts_all(img, self.resx, self.resy, self.current_warp, self.MModel))
        print 100*self.MModel+self.Algo
        self.J = np.asmatrix(sample_pts_Jacob(img, self.resx, self.resy, self.current_warp, 100*self.MModel))
        self.H_inv = (self.J.T * self.J).I
        if self.use_scv:
            self.intensity_map = np.arange(256, dtype=np.float64)
        self.initialized = True

    def initialize_with_rectangle(self, img, ul, lr):
        region_corners = \
            np.array([[ul[0], ul[1]],
                      [lr[0], ul[1]],
                      [lr[0], lr[1]],
                      [ul[0], lr[1]]], dtype=np.float64).T
        self.initialize(img, region_corners)

    def update(self, img):
        if not self.initialized: return
        img_old = img
        img = img.astype(np.uint8)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_img, img, self.pval, None, **self.lk_params)
        H, mask = cv2.findHomography(self.pval, p1, (0, cv2.RANSAC)[self.use_ransac], 10.0)
        self.current_warp = np.asmatrix(H) * self.current_warp
        temp_current_warp = self.current_warp.copy()
        self.prev_img = img
        for iter in xrange(self.max_iters):
            sampled_img = sample_pts_all(img_old, self.resx, self.resy, self.current_warp, self.MModel)
            if self.use_scv:
                if self.intensity_map == None: self.intensity_map = scv_intensity_map(sampled_img, self.template)
                sampled_img = scv_expected_img(sampled_img, self.intensity_map)
            error = np.asarray(sampled_img - self.template).reshape(-1,1)
            J = np.asmatrix(self.J[mask[:,0]>0, :])
            H_inv = (J.T * J) .I
            update = J.T * error[mask[:,0]>0]            
            update = H_inv * update
            update = np.asarray(update).squeeze()

            temp_matrix = np.empty((1,9),dtype=np.float64)
            temp_matrix[0,:8] = update[:]
            temp_matrix[0,8] = 1.0
            temp_matrix = temp_matrix.reshape(3,3)
            temp_matrix[0,0] += 1
            temp_matrix[1,1] += 1
            self.current_warp = np.asmatrix(self.current_warp) * pinv(temp_matrix)
            normalize_hom(self.current_warp)                                
            if np.sum(np.abs(update)) < self.threshold: break
        if self.use_scv:
            sampled_img = sample_pts(img_old, self.resx, self.resy, self.current_warp)
            self.intensity_map = scv_intensity_map(sampled_img, self.template)
        diff_val = apply_to_pts_all(temp_current_warp, np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T,1) - apply_to_pts_all(self.current_warp, np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T,1)
        diff_val = math.sqrt(np.sum(diff_val**2)/diff_val.shape[1])
        if diff_val > 15:
            self.current_warp = temp_current_warp
            normalize_hom(self.current_warp)
        self.p0 = apply_to_pts_all(self.current_warp, self.pos, 1)
        self.p0 = self.p0.T.reshape(self.resx*self.resy,1,2)
        self.p0 = self.p0.astype(np.float32)        
        self.pval = np.copy(self.p0)

    def set_intensity_map(self, intensity_map):
        pass
    
    def get_intensity_map(self):
        pass

    def is_initialized(self):
        return self.initialized

    def set_warp(self, warp, reset_intensity=True):
        self.current_warp = warp
        temp_x = np.linspace(-.5,.5,self.resx)
        temp_y = np.linspace(-.5,.5,self.resy)
        xv, yv = np.meshgrid(temp_x, temp_y)
        self.pos = np.empty((2,self.resx*self.resy),dtype=np.float32)
        self.pos[0,:] = xv.flatten()[:]
        self.pos[1,:] = yv.flatten()[:]
        self.p0 = apply_to_pts_all(self.current_warp, self.pos, 1)
        self.p0 = self.p0.T.reshape(self.resx*self.resy,1,2)
        self.p0 = self.p0.astype(np.float32)
        self.pval = np.copy(self.p0)
        if reset_intensity: self.intensity_map = None

    def get_warp(self):
        return np.asmatrix(self.current_warp)

    def set_region(self, corners, reset_intensity=True, img=None):
        self.current_warp = square_to_corners_warp(np.asarray(corners))
        temp_x = np.linspace(-.5,.5,self.resx)
        temp_y = np.linspace(-.5,.5,self.resy)
        xv, yv = np.meshgrid(temp_x, temp_y)
        self.pos = np.empty((2,self.resx*self.resy),dtype=np.float32)
        self.pos[0,:] = xv.flatten()[:]
        self.pos[1,:] = yv.flatten()[:]
        self.p0 = apply_to_pts_all(self.current_warp, self.pos, 1)
        self.p0 = self.p0.T.reshape(self.resx*self.resy,1,2)
        self.p0 = self.p0.astype(np.float32)
        self.pval = np.copy(self.p0)
        if img is not None: 
            self.prev_img = np.copy(img)
        else: self.prev_img = np.copy(self.origin_img)
        if reset_intensity: self.intensity_map = None

    def get_region(self):
        return apply_to_pts_all(self.get_warp(), np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T,1)        

    
