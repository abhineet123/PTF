""" 
Implementation of the Baker+Matthews Inverse Compositional Tracker.
Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import cv2

cimport numpy as np
import numpy as np
from utility cimport *
from imgtool import *
from imgtool cimport *
import time
import pdb

cdef class PFTracker:

    cdef:
        int n_samples
        int resx
        int resy
        np.ndarray template, template_old
        double alpha
        double threshold
        
        double[:,:] sz_T
        double[:,:] current_warp
        double[:] intensity_map
        bint use_scv
        bint initialized

        double[:,:] est
        double[:,:] affsig
        double[:,:] param

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

    def __init__(self, int n_samples, int resx, int resy, bint use_scv, int MModel =  1):
        self.n_samples = n_samples
        self.resx = resx
        self.resy = resy
        self.sz_T = np.empty((1,2),dtype=np.float64)
        self.sz_T[0,0] = self.resx
        self.sz_T[0,1] = self.resy
        self.MModel = MModel
        print 'Initializing Cython PF tracker with:'
        print 'n_samples: ', self.n_samples
        print 'resx: ', self.resx
        print 'resy: ', self.resy
        print 'use_scv: ', self.use_scv
        print 'MModel: ', self.MModel

        self.affsig = np.array([[9.,9.,0.04,0.04,0.005,0.001]],dtype=np.float64)
        #self.affsig = np.array([[0., 0., 0., 0., 0., 0.]], dtype=np.float64)
        self.alpha = 50.0
        self.use_scv = use_scv
        self.initialized = False
        self.current_warp = np.empty((0,0))
        #self.fobj = open('output.log','w')

    cpdef initialize(self, double[:,:] img, double[:,:] region_corners):
        cpdef double ctx, cty, wwidth, hheight
        self.initialized = False
        
        ctx = np.mean(region_corners, axis=1)[0]
        cty = np.mean(region_corners, axis=1)[1]
        wwidth = (region_corners[0,1]+region_corners[0,2]-region_corners[0,3]-region_corners[0,0]) / 2.0
        hheight = (region_corners[1,3]+region_corners[1,2]-region_corners[1,0]-region_corners[1,1]) / 2.0
        # dx, dy, sc, th, sr, phi        
        self.est = affinv(region_corners, self.sz_T) # Good
        # TODOed generate est, p1,p2,p3;p4,p5,p6
        self.param = np.array(np.tile(affparam2geom(self.est),[self.n_samples,1]))
        self.template = np.asarray(warpimg(img, self.est, self.sz_T)) #sz: 1*2 w,h !!!Wrong
        self.template_old = self.template.copy()
        self.intensity_map = np.arange(256, dtype=np.float64)
        self.template = np.asarray(whiten(self.template))
        self.template /= np.linalg.norm(self.template)        
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
        cdef double[:,:] samples
        cdef double[:] p
        cdef int n
        samples, self.param = estwarp_condens(img, self.param, self.n_samples, self.sz_T, self.affsig, self.MModel)
        wlist = weight_eval(samples, self.template, self.sz_T, self.alpha, self.use_scv, self.intensity_map, self.template_old)
        (q,indq) = des_sort(wlist)
        id_max = indq[0] #change here
        temp = np.zeros((1,6))
        temp[0,:] = self.param[id_max,:]
        self.est = affparam2mat(temp)
        p = np.zeros(self.n_samples,np.float64) #observation likelihood initialization
        n = 0        
        while (n<self.n_samples):
            if q[indq[n]] < 0:
                print('Prob should be positive')
                #sys.exit()
            p[indq[n]] = q[n]
            n += 1
        self.param = resample2(np.array(self.param), np.array(p))

        if self.use_scv:
            sampled_img = warpimg(img, self.est, self.sz_T) # TODO
            self.intensity_map = scv_intensity_map(sampled_img[:,0], self.template_old[:,0])
            

    cpdef set_intensity_map(self, double[:] intensity_map):
        self.intensity_map = intensity_map

    cpdef double[:] get_intensity_map(self):
        return self.intensity_map

    cpdef is_initialized(self):
        return self.initialized

    cpdef set_warp(self, double[:,:] warp, bint reset_intensity=True):
        self.est = warp
        if reset_intensity: self.intensity_map = None

    cpdef double[:,:] get_warp(self):
        return np.asmatrix(self.est)

    cpdef double[:,:] get_tmplt(self):
        return np.asarray(self.template.reshape(self.resx,self.resy))

    cpdef set_region(self, double[:,:] corners, bint reset_intensity=True):

        ctx = np.mean(corners, axis=1)[0]
        cty = np.mean(corners, axis=1)[1]
        wwidth = (corners[0,1]+corners[0,2]-corners[0,3]-corners[0,0]) / 2.0
        hheight = (corners[1,3]+corners[1,2]-corners[1,0]-corners[1,1]) / 2.0
        # dx, dy, sc, th, sr, phi
        self.est = affinv(corners, self.sz_T)
        for indx in range(0,10,self.param.shape[0]):
            self.param[indx,:] = self.est[0,:]
        if reset_intensity: self.intensity_map = None

    cpdef get_region(self):
        return drawbox(self.est, self.sz_T)
    

