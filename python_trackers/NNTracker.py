"""
Implementation of the Nearest Neighbour Tracking Algorithm.
Author: Travis Dick (travis.barry.dick@gmail.com)
"""
# import msvcrt
from SCVUtils import *
from TrackerBase import *
from FeatureUtils import *
from WarpIndex import *
import numpy as np
import pyflann
# from scipy import weave
# from scipy.weave import converters
from Homography import *
from ImageUtils import *
import itertools
import operator

import cv2
# Jesse
#import pdb
#import sys


class NNTracker(TrackerBase):
    def __init__(self, no_of_samples, no_of_iterations=1, res=(20, 20), multi_approach='none',
                 warp_generator=lambda: random_homography(0.07, 0.06),
                 use_scv=False, feature='none'):
        """ An implemetation of the Nearest Neighbour Tracker. 

        Parameters:
        -----------
        n_samples : integer
          The number of sample motions to generate. Higher values will improve tracking
          accuracy but increase running time.
        
        n_iterations : integer
          The number of times to update the tracker state per frame. Larger numbers
          may improve convergence but will increase running time.
        
        res : (integer, integer)
          The desired resolution of the template image. Higher values allow for more
          precise tracking but increase running time.

        warp_generator : () -> (3,3) numpy matrix.
          A function that randomly generates a homography. The distribution should
          roughly mimic the types of motions that you expect to observe in the 
          tracking sequence. random_homography seems to work well in most applications.
          
        See Also:
        ---------
        TrackerBase
        BakerMatthewsICTracker
        """

        self.resx = res[0]
        self.resy = res[1]
        if self.resy <= 0:
            self.resy = self.resx
        self.res = (self.resx, self.resy)

        print "Initializing NN tracker with:"
        print " n_samples=", no_of_samples
        print " n_iterations=", no_of_iterations
        print " res=", self.res
        print " multi_approach=", multi_approach
        print " use_scv=", use_scv
        print " feature=", feature

        self.n_samples = no_of_samples
        self.n_iterations = no_of_iterations
        self.multi_approach = multi_approach

        self.warp_generator = warp_generator
        self.n_points = np.prod(res).astype(np.int32)
        self.initialized = False
        self.proposal = None
        self.last_updates = []
        self.use_las_update = True
        self.pts = res_to_pts(self.res)
        self.use_scv = use_scv
        self.sift = False
        self.unit_square = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]).T
        self.feature = feature
        self.feature_obj = getFeatureObject(self.feature, self.multi_approach)
        self.n_channels = 1

        self.flatten = False
        self.use_mean = False

        self.warp_index = None
        self.template = None

        if self.multi_approach == 'none':
            print 'Using single channel approach'
            self.initialize = self.initializeSingleChannel
            self.update = self.updateSingleChannel
        else:
            print 'Using multi channel approach'
            self.initialize = self.initializeMultiChannel
            self.update = self.updateMultiChannel
            if self.multi_approach == 'flatten':
                print 'Using flattening'
                self.flatten = True
            else:
                if self.multi_approach == 'mean':
                    print 'Using mean'
                    self.use_mean = True
                else:
                    print 'Using ' + self.multi_approach

    def initializeMultiChannel(self, img, region):
        #print "starting nn initialize"

        self.set_region(region)

        img_shape = img.shape
        #print "img.shape=",img_shape
        if len(img_shape) < 3:
            raise SystemExit('Error in NNTracker: '
                             'Expected multi channel image but found single channel one')
        else:
            self.n_channels = img_shape[2]

        self.feature_obj.initialize(self.n_channels)
        self.feature_obj.updateSCVTemplate(img, self.pts, self.get_warp())

        #self.use_scv=False
        self.sift = False
        self.template = self.feature_obj.getFeature(img, self.pts, self.get_warp())
        #print 'Done sampling'
        self.warp_index = WarpIndexVec(self.n_samples, self.warp_generator, img, self.pts,
                                       self.n_channels, self.get_warp(), self.res,
                                       self.multi_approach, feature_obj=self.feature_obj)
        #print "Done creating warp index"
        self.initialized = True
        #print "done"

    def updateMultiChannel(self, img, old_region=None):
        #print "starting nn update"
        if not self.is_initialized():
            return None

        if self.use_las_update:
            for update in self.last_updates:
                self.proposal = self.proposal * update
                self.proposal /= self.proposal[2, 2]

        self.last_updates = []
        for i in xrange(self.n_iterations):
            #warped_pts = apply_to_pts(self.proposal, self.pts)
            sampled_img = self.feature_obj.getFeature(img, self.pts, warp=self.proposal,
                                                      use_scv=self.use_scv)

            update = self.warp_index.getBestMatch(sampled_img)
            self.proposal = self.proposal * update
            self.proposal /= self.proposal[2, 2]
            self.last_updates.append(update.copy())
        #print "done"
        if self.use_scv:
            self.feature_obj.updateSCVIntensityMap(img, self.pts, self.get_warp())

        return self.proposal

    def initializeSingleChannel(self, img, region):
        #print "starting nn initialize"
        self.set_region(region)

        img_shape = img.shape
        #print "img.shape=",img_shape
        if len(img_shape) != 2:
            raise SystemExit('Error in NNTracker: '
                             'Expected single channel image but found multi channel one')

        self.feature_obj.updateSCVTemplate(img, self.pts, self.get_warp())
        self.template = self.feature_obj.getFeature(img, self.pts, self.get_warp())
        self.warp_index = WarpIndex(self.n_samples, self.warp_generator, img, self.pts,
                                    self.get_warp(), self.res, self.feature_obj)
        #Jesse
        #pdb.set_trace()
        self.initialized = True
        #print "done"

    def updateSingleChannel(self, img):
        #print "starting nn update"
        if not self.is_initialized():
            return None

        if self.use_las_update:
            for update in self.last_updates:
                self.proposal = self.proposal * update
                self.proposal /= self.proposal[2, 2]

        self.last_updates = []
        for i in xrange(self.n_iterations):
            #warped_pts = apply_to_pts(self.proposal, self.pts)
            sampled_img = self.feature_obj.getFeature(img, self.pts, warp=self.proposal,
                                                      use_scv=self.use_scv)

            if not self.sift:
                update = self.warp_index.getBestMatch(sampled_img)
            else:
                # --sift-- #
                temp_desc = self.pixel2sift(sampled_img)
                #	if temp_desc == None:
                #		print('No feature found!')
                #		sys.exit()
                #       pdb.set_trace()
                update = self.desc2warp_weighted3(temp_desc)
            self.proposal = self.proposal * update
            self.proposal /= self.proposal[2, 2]
            self.last_updates.append(update.copy())

        if self.use_scv:
            self.feature_obj.updateSCVIntensityMap(img, self.pts, self.get_warp())

        #print "done"
        return self.proposal

    def is_initialized(self):
        return self.initialized

    def get_warp(self):
        return self.proposal

    def set_region(self, corners):
        # if self.proposal is not None:
        #     self.old_proposal = np.copy(self.proposal)
        self.proposal = square_to_corners_warp(corners)

    def get_region(self):
        return apply_to_pts(self.get_warp(), self.unit_square)

    def cleanup(self):
        pass

        #-- sift --#

    def pixel2sift(self, patch):
        detector = cv2.FeatureDetector_create("SIFT")
        detector.setDouble('edgeThreshold', 30)
        descriptor = cv2.DescriptorExtractor_create("SIFT")
        #sift = cv2.SIFT(edgeThreshold = 20)
        patch = (patch.reshape(self.resx, self.resy)).astype(np.uint8)
        skp = detector.detect(patch)
        skp, sd = descriptor.compute(patch, skp)
        #pdb.set_trace()
        #print(sd.shape[0])
        return sd

    # --- For sift --- #
    def desc2warp_weighted(self, descs):
        warps = np.zeros((3, 3), dtype=np.float64)
        temp_desc = np.empty((128, 1), dtype=np.float32)
        if descs == None:
            print('The number of descriptors is zero!')
            return np.eye(3, dtype=np.float32)
        for i in range(descs.shape[0]):
            temp_desc[:, 0] = descs[i, :]
            warp, dist = self.warp_index.best_match_sift(temp_desc.T)
            warps += warp
        warps /= descs.shape[0]
        return warps

    # --- For sift --- #
    def desc2warp_weighted2(self, descs):
        warps = np.zeros((3, 3), dtype=np.float64)
        temp_desc = np.empty((128, 1), dtype=np.float32)
        if descs == None:
            print('The number of descriptors is zero!')
            return np.eye(3, dtype=np.float32)
        warp_list = []
        dist_list = []
        for i in range(descs.shape[0]):
            temp_desc[:, 0] = descs[i, :]
            warp, dist = self.warp_index.best_match_sift(temp_desc.T)
            #warps += warp
            warp_list.append(warp)
            dist_list.append(dist)
        thres = max(dist_list) * 0.5
        count = 0
        for i in range(len(dist_list)):
            if dist_list[i] <= thres:
                warps += warp_list[i]
                count += 1
        if count == 0: return np.eye(3, dtype=np.float32)
        warps /= count
        return warps

    # --- For sift --- #
    def desc2warp_weighted3(self, descs):
        warps = np.zeros((3, 3), dtype=np.float64)
        temp_desc = np.empty((128, 1), dtype=np.float32)
        if descs == None:
            print('The number of descriptors is zero!')
            return np.eye(3, dtype=np.float32)
        warp_list = []
        dist_list = []
        print('Testing')
        for i in range(descs.shape[0]):
            temp_desc[:, 0] = descs[i, :]
            warp, dist, index = self.warp_index.best_match_sift(temp_desc.T)
            print(index)
            #warps += warp
            warp_list.append(warp)
            dist_list.append(dist)
        sum_dist = sum(dist_list)
        for i in range(len(dist_list)):
            warps += warp_list[i] * dist_list[i] / sum_dist
        return warps
