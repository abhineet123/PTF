"""
Implementation of the Baker-Matthews Inverse Compositional Tracking Algorithm.

S. Baker and I. Matthews, "Equivalence and efficiency of image alignment algorithms", 
Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE 
Computer Society Conference on, vol. 1, pp. I-1090-I-1097 vol. 1, 2001.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import numpy as np
from Homography import *
from SCVUtils import *
from TrackerBase import *
from ImageUtils import *
from FeatureUtils import *


class BakerMatthewsICTracker(TrackerBase):
    def __init__(self, max_iterations, threshold=0.01, res=(20, 20), multi_approach='none',
                 use_scv=False, feature='none'):
        """ An implementation of the inverse composititionl tracker from Baker and Matthews.

        Parameters:
        -----------
        max_iters : integer
          The maximum number of iterations per frame

        threshold : real
          If the decrease in error is smaller than the threshold then the per-frame
          update terminates.

        res : (integer, integer)
          The desired resolution of the template image. Higher values allow for more
          precise tracking but increase running time.
        
        See Also:
        ---------
        TrackerBase
        NNTracker
        """

        resx = res[0]
        resy = res[1]
        if resy <= 0:
            resy = resx
        self.res = (resx, resy)

        print "Initializing ICT tracker with:"
        print " max_iters=", max_iterations
        print " threshold=", threshold
        print " res=", self.res
        print " multi_approach=", multi_approach
        print 'use_scv=', use_scv
        print " feature=", feature

        self.max_iters = max_iterations
        self.pts = res_to_pts(self.res)
        self.n_pts = np.prod(res)
        self.initialized = False
        self.use_scv = use_scv
        self.threshold = threshold
        self.multi_approach = multi_approach
        self.n_channels = 1
        self.proposal = None

        self.VT_dW_dp = None
        self.template = None
        self.H_inv = None

        self.unit_square = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]).T
        self.feature_obj = getFeatureObject(feature, self.multi_approach)

        self.flatten = False
        self.use_mean = False
        self.count = 0

        self.current_corners = None
        self.last_corners = self.unit_square

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

    def initializeMultiChannel(self, img, region):
        # print "in ICTTracker initialize"

        img_shape = img.shape
        if len(img_shape) < 3:
            raise SystemExit('Error in BakerMatthewsICTracker: '
                             'Expected multi channel image but found single channel one')
        else:
            self.n_channels = img_shape[2]

        self.set_region(region)
        self.feature_obj.initialize(self.n_channels)
        self.feature_obj.updateSCVTemplate(img, self.pts, self.proposal)
        self.template = self.feature_obj.getFeature(img, self.pts, self.get_warp())
        self.VT_dW_dp = self.feature_obj.getJacobian(img, self.pts, self.get_warp())
        if self.flatten:
            H = self.VT_dW_dp.T * self.VT_dW_dp
            self.H_inv = H.I
        else:
            self.H_inv = None
            for i in xrange(self.n_channels):
                VT_dW_dp_ch = np.asmatrix(self.VT_dW_dp[i, :, :])
                H = VT_dW_dp_ch.T * VT_dW_dp_ch
                if self.H_inv is None:
                    self.H_inv = np.empty((self.n_channels, ) + H.shape)
                self.H_inv[i, :, :] = H.I
        self.initialized = True

    def updateMultiChannel(self, img):
        if not self.is_initialized():
            return None
        # if use_update is not None:
        # # print 'provided update:\n', use_update
        # self.proposal=use_update
        # if self.current_corners is not None:
        # self.last_corners=self.current_corners.copy()
        # self.current_corners=self.get_region()
        #     # self.proposal = self.proposal * use_update
        #     return
        for i in xrange(self.max_iters):
            IWxp = self.feature_obj.getFeature(img, self.pts, self.get_warp(),
                                               use_scv=self.use_scv)
            #np.savetxt('IWxp.txt', IWxp.T, fmt='%12.6f', delimiter='\t')
            #np.savetxt('template.txt', self.template.T, fmt='%12.6f', delimiter='\t')
            error_img = np.asmatrix(IWxp - self.template)
            #np.savetxt('error_img.txt', error_img.T, fmt='%12.6f', delimiter='\t')
            if self.flatten:
                update = np.asmatrix(self.VT_dW_dp.T) * error_img.reshape((-1, 1))
                update = self.H_inv * np.asmatrix(update).reshape((-1, 1))
                update = np.asarray(update).squeeze()
            else:
                update_sum = np.zeros(8)
                for i in xrange(self.n_channels):
                    #np.savetxt('err.txt', err.T, fmt='%12.6f', delimiter='\t')
                    #np.savetxt('VT_dW_dp.txt', VT_dW_dp, fmt='%12.6f', delimiter='\t')
                    update = np.asmatrix(self.VT_dW_dp[i, :, :].T) * error_img[i, :].reshape((-1, 1))
                    update = self.H_inv[i, :, :] * np.asmatrix(update).reshape((-1, 1))
                    update = np.asarray(update).squeeze()
                    update_sum += update
                    #    print "update ",i, ":\n", update
                #print "update_vec:\n", update_vec
                update = update_sum / self.n_channels
            # update_hom=make_hom_sl3(update).I
            # print 'computed update:\n', update_hom
            self.proposal = self.proposal * make_hom_sl3(update).I
        if self.use_scv:
            self.feature_obj.updateSCVIntensityMap(img, self.pts, self.get_warp())


    def initializeSingleChannel(self, img, region):
        # print "in ICTTracker initialize"

        img_shape = img.shape
        if len(img_shape) != 2:
            raise SystemExit('Error in BakerMatthewsICTracker: '
                             'Expected single channel image but found multi channel one')

        self.set_region(region)

        self.feature_obj.updateSCVTemplate(img, self.pts, self.proposal)
        self.template = self.feature_obj.getFeature(img, self.pts, self.get_warp())
        self.VT_dW_dp = self.feature_obj.getJacobian(img, self.pts, self.get_warp())
        H = self.VT_dW_dp.T * self.VT_dW_dp
        self.H_inv = H.I
        self.initialized = True

        # Image Gradient:
        # nabla_T = image_gradient(img, self.pts, self.get_warp())
        # Steepest Descent Images:
        # self.VT_dW_dp = np.empty((self.n_pts, 8))
        # for i in xrange(self.n_pts):
        # self.VT_dW_dp[i,:] = np.asmatrix(nabla_T[i,:]) * _make_hom_jacobian(self.pts[:,i])
        #self.VT_dW_dp = np.asmatrix(self.VT_dW_dp)

        # Hessian:

        # H = np.zeros((8,8))
        # for i in xrange(self.n_pts):
        #     H += np.asmatrix(self.VT_dW_dp[i,:].T) * self.VT_dW_dp[i,:]
        # self.H_inv = np.asmatrix(H).I

    def updateSingleChannel(self, img):
        if not self.is_initialized():
            return None

        # if use_update is not None:
        # self.proposal = self.proposal * use_update
        # return

        for i in xrange(self.max_iters):
            IWxp = self.feature_obj.getFeature(img, self.pts, self.get_warp(),
                                               use_scv=self.use_scv)
            # if self.use_scv:
            # IWxp = scv_expectation(IWxp, self.template)
            error_img = np.asmatrix(IWxp - self.template)
            update = np.asmatrix(self.VT_dW_dp.T) * error_img.reshape((-1, 1))
            update = self.H_inv * np.asmatrix(update).reshape((-1, 1))
            update = np.asarray(update).squeeze()
            self.proposal = self.proposal * make_hom_sl3(update).I
            # if np.sum(np.abs(update)) < self.threshold:
        if self.use_scv:
            self.feature_obj.updateSCVIntensityMap(img, self.pts, self.get_warp())

    def is_initialized(self):
        return self.initialized

    def get_warp(self):
        return self.proposal

    def set_region(self, corners):
        self.proposal = square_to_corners_warp(corners)

    def get_region(self):
        return apply_to_pts(self.get_warp(), self.unit_square)

    def cleanup(self):
        pass

    def getMultiApproach(self):
        return self.multi_approach

    def getSCVState(self):
        return self.use_scv


# import numpy as np
# from scipy.linalg import expm

# def _make_hom(p):
# return np.matrix([[1 + p[0],   p[1]  , p[2]],
# [  p[3]  , 1 + p[4], p[5]],
# [  p[6]  ,   p[7]  ,   1]], dtype=np.float64)

# def _make_hom_jacobian(pt, p=np.zeros(8)):
# x = pt[0]
# y = pt[1]
#     jacobian = np.zeros((2,8))
#     d = (p[6]*x + p[7]*y + 1)
#     d2 = d*d
#     jacobian[0,0] = x / d
#     jacobian[0,1] = y / d
#     jacobian[0,2] = 1 / d
#     jacobian[1,3] = x / d
#     jacobian[1,4] = y / d
#     jacobian[1,5] = 1 / d
#     jacobian[0,6] = -x * (p[0]*x + x + p[1]*y + p[2]) / d2
#     jacobian[0,7] = -y * (p[0]*x + x + p[1]*y + p[2]) / d2
#     jacobian[1,6] = -x * (p[3]*x + p[4]*y + y + p[5]) / d2
#     jacobian[1,7] = -y * (p[3]*x + p[4]*y + y + p[5]) / d2
#     return np.asmatrix(jacobian)
