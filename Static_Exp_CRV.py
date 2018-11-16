"""
Implementation of the Static Image experiment found in Baker + Matthews 2004
Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import math

import cv2
import numpy as np
from numpy.linalg import inv

import shelve
import os
import scipy
import pdb

from cython_trackers.utility import *
from cython_trackers.TurnkeyTrackers import *

_select = 'static'
if _select == 'static':
    _stored_warps = shelve.open(os.path.expanduser("~/static_cache1"))
    _warps = _stored_warps['1-20 5000']

class StaticImageExperiment:

    def __init__(self, image, target_region, sigmas, n_trials, max_spatial_sigma, single_template=False):

        self.image = image
        self.target_region = target_region
        self.sigmas = sigmas
        self.n_trials = n_trials
        self.max_spatial_sigma = max_spatial_sigma
        self.single_template = single_template
        self.files = []    
    
    def run(self, trackers):
        if self.single_template:
            for tracker in trackers: 
                tracker.initialize(self.image, self.target_region)
                initial_warp = tracker.get_warp()
                if _towrite:
                    path = _write_path + '/' + _log[trackers.index(tracker)]+'.txt'
                    self.files.append(open(path,'w'))
        results = []
        # size
        height, width = img.shape
        xwidth = np.linspace(1,width,width)
        xheight = np.linspace(1,height,height)
        xpos, ypos = scipy.meshgrid(xwidth, xheight)
        xy_pts = np.empty((2,width*height), dtype=np.float64)
        xy_pts[0,:] = xpos.flatten()
        xy_pts[1,:] = ypos.flatten()
        # apply_to_pts, pts is 2*n
        indx = 1
        for sigma in self.sigmas:
            n_converged = [0 for x in range(len(trackers))]
            for i in xrange(self.n_trials):
                if _select == 'dynamic':
                    point_noise = np.random.normal(0, sigma, (2,4))
                    disturbed_region = self.target_region + point_noise
                elif _select == 'static':
                    disturbed_region = _warps[int(sigma)-1,i, :, :]    
                # size
                Homo_imgs = compute_homography(self.target_region, disturbed_region)
                # We need to find inverse of Homo instead of Homo_imgs
                Homo_imgs = inv(Homo_imgs)
                #pdb.set_trace()
                new_pts = apply_to_pts(Homo_imgs, xy_pts)
                temp_x = np.array(new_pts[0,:]).reshape(height, width)
                temp_y = np.array(new_pts[1,:]).reshape(height, width)
                new_img = sample_pts_raw(self.image, temp_x, temp_y)
                new_img = np.asarray(new_img).reshape(height, width)
                if not self.single_template:
                    tracker.initialize(self.image, disturbed_region)
                    tracker.set_warp(initial_warp)
                else:
                    for tracker in trackers:
                        tracker.set_region(self.target_region)
                for tracker in trackers:
                    tracker.update(new_img)

                if not self.single_template:
                    true_region = self.target_region
                else:
                    true_region = disturbed_region
                rgb_img = cv2.cvtColor(new_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)    
                #pdb.set_trace()
                for index in range(len(trackers)):
                    draw_region(rgb_img,trackers[index].get_region(),  _color[index], _thickness[index])
                #cv2.imwrite('icra_res/tracked_%05d.jpg'%(indx), rgb_img)
                #cv2.imshow('J', rgb_img)
                #cv2.waitKey(10)
                indx += 1
                for tracker in trackers:
                    if _point_rmse(true_region, tracker.get_region()) < self.max_spatial_sigma:
                        n_converged[trackers.index(tracker)] += 1
                    if _towrite:
                        fobj = self.files[trackers.index(tracker)]
                        fobj.write(region_to_string(tracker.get_region()) + "\n")
            temp_res = []
            for trk_indx in range(len(trackers)):
                convergence_rate = n_converged[trk_indx] / float(self.n_trials)
                print "Sigma %g : %g" % (sigma, convergence_rate)
                temp_res.append(convergence_rate)
            results.append(temp_res)
        for fobj in self.files:
            fobj.close()
        return results
def region_to_string(region):
    output = ""
    for i in xrange(0,4):
        for j in xrange(0,2):
            output += "\t%.2f"%region[j,i]
    return output

def _point_rmse(a,b):
    return  math.sqrt(np.sum((a-b)**2)/a.shape[1])

# Set up an experiment with the lena image. This is tedious to retype every time.
# TODO: Remove this before publishing code.
img = cv2.resize(np.asarray(to_grayscale(cv2.imread("./lena.jpg"))), (256, 256))
img = cv2.GaussianBlur(img, (3,3), 0.75)
ul = (256/2-50, 256/2-50)
lr = (256/2+50, 256/2+50)
region = rectangle_to_region(ul, lr)
sigmas = np.arange(1,20.1,1)
experiment = StaticImageExperiment(img, region, sigmas, 5000, 3, True)
# Making trackers
nnic_tracker = make_nn_bmic(use_scv=True, res=(50,50), nn_samples=4000, bmic_iters=30)
trackers = []
trackers.append(nnic_tracker)
color = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
_thickness = [5,4,3,2]
_write_path = '/home/xzhang6/Documents/fuerte_ws/sandbox/thesis_results/CRV/Static_exp_new'
# TODO we use static_exp_new instead of static_exp_new2
_towrite = True
log = ['nnic']
experiment.run(trackers)
