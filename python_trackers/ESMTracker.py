"""
Implementation of the ESM tracking algorithm.

S. Benhimane and E. Malis, "Real-time image-based tracking of planes
using efficient second-order minimization," Intelligent Robots and Systems, 2004.
(IROS 2004). Proceedings. 2004 IEEE/RSJ International Conference on, vol. 1, 
pp. 943-948 vol. 1, 2004.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import numpy as np
from scipy.linalg import expm

from Homography import *
from ImageUtils import *
from TrackerBase import *
import os
from FeatureUtils import *


class ESMTracker(TrackerBase):
    def __init__(self, max_iterations, threshold=0.01, err_thresh=175, res=(20, 20), multi_approach='none',
                 use_err_thresh=True, use_scv=True, feature='none'):

        resx = res[0]
        resy = res[1]
        if resy <= 0:
            resy = resx
        self.res = (resx, resy)

        print "Initializing ESM tracker with:"
        print "\t max_iters=", max_iterations
        print "\t threshold=", threshold
        print "\t err_thresh=", err_thresh
        print "\t res=", self.res
        print "\t multi_approach=", multi_approach
        print "\t use_err_thresh=", use_err_thresh
        print "\t use_scv=", use_scv

        self.max_iters = max_iterations
        self.threshold = threshold
        self.npts = np.prod(res)
        self.pts = res_to_pts(self.res)
        self.use_scv = use_scv
        self.initialized = False
        self.mean_level = 0
        self.unit_square = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]).T
        self.multi_approach = multi_approach
        self.use_err_thresh=use_err_thresh
        self.err_thresh = err_thresh

        self.n_channels = 1
        self.template = None
        self.proposal = None
        self.Je = None
        self.feature_obj = getFeatureObject(feature, self.multi_approach)

        self.flatten = False
        self.use_mean = False
        # proposal_file_name = 'proposal.txt'
        # if os.path.exists(proposal_file_name):
        # os.remove(proposal_file_name)
        # self.proposal_file = open(proposal_file_name, 'a')
        self.frame_count = 0

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
        # print "in ESMTracker initialize"
        img_shape = img.shape
        if len(img_shape) != 3:
            raise SystemExit('Error in ESMTracker:'
                             'Expected multi channel image but found single channel one instead')

        self.n_channels = img_shape[2]
        self.set_region(region)

        self.feature_obj.initialize(self.n_channels)
        self.feature_obj.updateSCVTemplate(img, self.pts, self.proposal)
        self.template = self.feature_obj.getFeature(img, self.pts, self.proposal)
        self.Je = self.feature_obj.getJacobian(img, self.pts, self.proposal)

        self.initialized = True
        # print "Done"

    def updateMultiChannel(self, img):

        # self.frame_count += 1
        # print 'frame_count=', self.frame_count
        # self.proposal_file.write('frame %d\n'%self.frame_count)

        # print "in ESMTracker update"
        if not self.initialized:
            return
        # update_thresh=False

        for i in xrange(self.max_iters):
            # if update_thresh:
            # break
            # update_sum=np.zeros(8)
            sampled_img = self.feature_obj.getFeature(img, self.pts, self.get_warp(),
                                                      use_scv=self.use_scv)
            error_img = np.asmatrix(self.template - sampled_img)
            Jpc = self.feature_obj.getJacobian(img, self.pts, self.proposal)
            J = (Jpc + self.Je) / 2.0

            if self.flatten:
                # print "J shape=", J.shape
                update = np.asarray(np.linalg.lstsq(J, error_img.reshape((-1, 1)))[0]).squeeze()
                # print 'update.shape=', update.shape
                #update=self.getUpdate(img, self.template, self.Je)
                # self.writeUpdateData(update)
                # if np.isnan(self.proposal).any() or np.isinf(self.proposal).any():
                #     self.writeData(Jpc, J, update, error_img, sampled_img)
                #     raise SystemExit('Error in updateMultiChannel:\t'
                #                      'Encountered invalid warp update')
            else:
                # update_vec=[]
                update_sum = np.zeros(8)
                for ch in xrange(self.n_channels):
                    # print "J shape=", J.shape
                    #print 'error.shape=', error_img.shape
                    # print 'Getting Solution'
                    update_ch = np.asarray(
                        np.linalg.lstsq(np.asmatrix(J[ch, :, :]), error_img[ch, :].reshape((-1, 1)))[0]).squeeze()
                    # print 'Done getting solution'
                    update_sum += update_ch
                    # print 'update_ch=\n', update_ch

                    #update_vec.append(update)
                    #print 'update.shape=', update.shape
                    #print 'proposal.shape=', self.proposal.shape
                    #update=self.getUpdate(img[:, :, i], self.template_vec[i], self.Je_vec[i])

                # mean_update=getMean(update_vec)
                update = update_sum / self.n_channels
                # print 'update_sum=\n', update_sum
                # print 'update=\n', update
            self.proposal = self.proposal * make_hom_sl3(update)
            if np.sum(np.abs(update)) < self.threshold:
                break
        if self.use_scv:
            self.feature_obj.updateSCVIntensityMap(img, self.pts, self.get_warp())
            # self.proposal_file.write('-'*50+'\n')

    def initializeSingleChannel(self, img, region):
        # print "in ESMTracker initialize"
        img_shape = img.shape
        if len(img_shape) != 2:
            raise SystemExit('Error in ESMTracker: '
                             'Expected single channel image but found multi channel one')

        self.set_region(region)

        self.feature_obj.updateSCVTemplate(img, self.pts, self.proposal)
        # print "Sampling template"
        self.template = self.feature_obj.getFeature(img, self.pts, self.get_warp())
        # print "estimating jacobian"
        self.Je = self.feature_obj.getJacobian(img, self.pts, self.proposal)

        self.initialized = True
        # print "Done"

    def updateSingleChannel(self, img):
        # self.frame_count += 1
        # self.proposal_file.write('frame %d\n' % self.frame_count)
        # print "in ESMTracker update"
        if not self.initialized:
            return
        for i in xrange(self.max_iters):
            sampled_img = self.feature_obj.getFeature(img, self.pts, self.get_warp(),
                                                      use_scv=self.use_scv)
            error = np.asmatrix(self.template - sampled_img).reshape(-1, 1)
            if(self.use_err_thresh):
                error = error.squeeze()
                thresh_ids = np.array(error < self.err_thresh)
                thresh_ids = thresh_ids.squeeze().astype(np.bool)
                # print 'self.pts.shape: ', self.pts.shape
                # print 'thresh_ids.shape: ', thresh_ids.shape
                # print 'thresh_ids.dtype: ', thresh_ids.dtype
                #
                # print 'self.pts: ', self.pts
                # print 'thresh_ids: ', thresh_ids
                #
                # print 'before: error: ', error
                # print 'error.shape: ', error.shape

                error = error[0, thresh_ids].transpose()

                # print 'error.shape: ', error.shape
                # print 'after: error: ', error

                thresh_pts = self.pts[:, thresh_ids]

                # print 'thresh_pts: ', thresh_pts
                print 'thresh_pts.shape: ', thresh_pts.shape

                thresh_Jpc = self.feature_obj.getJacobian(img, thresh_pts, self.proposal)
                thresh_Je = self.Je[thresh_ids, :]

                J = (thresh_Jpc + thresh_Je) / 2.0
            else:
                Jpc = self.feature_obj.getJacobian(img, self.pts, self.proposal)
                J = (Jpc + self.Je) / 2.0
            # print "J shape=", J.shape

            update = np.asarray(np.linalg.lstsq(J, error)[0]).squeeze()

            self.proposal = self.proposal * make_hom_sl3(update)
            # self.writeUpdateData(update)
            if np.sum(np.abs(update)) < self.threshold:
                break
                # print "J_size=",J.shape
                # print "error_size=",error.shape
                # print "update_size=",update.shape
        if self.use_scv:
            self.feature_obj.updateSCVIntensityMap(img, self.pts, self.get_warp())

            # self.proposal_file.write('-' * 50 + '\n')

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
        # self.proposal_file.close()

        # def writeUpdateData(self, update):
        # self.proposal_file.write('update:\n')
        # np.savetxt(self.proposal_file, update, fmt='%12.6f', delimiter='\t')
        # self.proposal_file.write('proposal:\n')
        # np.savetxt(self.proposal_file, self.proposal, fmt='%12.6f', delimiter='\t')
        #
        # def writeData(self, Jpc, J, update, error_img, sampled_img):
        # if os.path.exists('error_data.txt'):
        # os.remove('error_data.txt')
        #     error_data_file = open('error_data.txt', 'a')
        #     error_data_file.write('\nJpc:\n')
        #     np.savetxt(error_data_file, Jpc, fmt='%12.6f', delimiter='\t')
        #     error_data_file.write('\nJ:\n')
        #     np.savetxt(error_data_file, J, fmt='%12.6f', delimiter='\t')
        #     error_data_file.write('\nupdate:\n')
        #     np.savetxt(error_data_file, update, fmt='%12.6f', delimiter='\t')
        #     error_data_file.write('\nupdate_hom:\n')
        #     np.savetxt(error_data_file, make_hom_sl3(update), fmt='%12.6f', delimiter='\t')
        #     error_data_file.write('\nself.proposal:\n')
        #     np.savetxt(error_data_file, self.proposal, fmt='%12.6f', delimiter='\t')
        #     error_data_file.close()
        #     np.savetxt('error_img.txt', error_img.T, fmt='%12.6f', delimiter='\t')
        #     np.savetxt('sampled_img.txt', sampled_img, fmt='%12.6f', delimiter='\t')
        #     print 'Jpc:\n', Jpc
        #     print 'J:\n', J
        #     print 'update:\n', update
        #     print 'self.proposal:\n', self.proposal

        # def _estimate_jacobian(img, pts, initial_warp, eps=1e-10):

        #    #print '_estimate_jacobian'
        #    n_pts = pts.shape[1]
        #    def f(p):
        #        W = initial_warp * make_hom_sl3(p)
        #        return sample_region(img, pts, W)
        #    jacobian = np.empty((n_pts,8))
        #    for i in xrange(0,8):
        #        o = np.zeros(8)
        #        o[i] = eps
        #        jacobian[:,i] = (f(o) - f(-o)) / (2*eps)
        #    return np.asmatrix(jacobian)
        #
        #def _estimate_jacobian_vec(img, pts, initial_warp, eps=1e-10):
        #    #print '_estimate_jacobian_vec'
        #    n_pts = pts.shape[1]*img.shape[2]
        #
        #    def f(p):
        #        W = initial_warp * make_hom_sl3(p)
        #        return sample_region_vec(img, pts, W, flatten=True)
        #
        #    jacobian = np.empty((n_pts, 8))
        #    for i in xrange(0, 8):
        #        o = np.zeros(8)
        #        o[i] = eps
        #        jacobian[:, i] = (f(o) - f(-o)) / (2 * eps)
        #    return np.asmatrix(jacobian)

        #def updateMultiChannel(self, img):
        #    #print "in ESMTracker update"
        #    if not self.initialized:
        #        return
        #
        #    #update_thresh=False
        #    for i in xrange(self.max_iters):
        #        #if update_thresh:
        #        #    break
        #        #update_sum=np.zeros(8)
        #        if self.flatten:
        #            update=self.getUpdate(img, self.template, self.Je)
        #            self.proposal = self.proposal * make_hom_sl3(update)
        #            if np.sum(np.abs(update)) < self.threshold:
        #                break
        #        else:
        #            update_vec=[]
        #            update_hom_vec=[]
        #            proposal_vec=[]
        #            pts_vec=[]
        #            thresh_count=0
        #            for i in xrange(self.n_channels):
        #                update=self.getUpdate(img[:, :, i], self.template_vec[i], self.Je_vec[i])
        #                #if np.sum(np.abs(update)) < self.threshold:
        #                #    thresh_count+=1
        #                if self.mean_level>0:
        #                    update_hom=make_hom_sl3(update)
        #                    if self.mean_level>1:
        #                        proposal=self.proposal * update_hom
        #                        if self.mean_level>2:
        #                            pts_vec.append(apply_to_pts(proposal, self.unit_square))
        #                            #print "pts_vec:\n", pts_vec
        #                        else:
        #                            proposal_vec.append(proposal)
        #                    else:
        #                        update_hom_vec.append(update_hom)
        #                else:
        #                    update_vec.append(update)
        #            if self.mean_level==0:
        #                mean_update=getMean(update_vec)
        #                self.proposal = self.proposal * make_hom_sl3(mean_update)
        #                if np.sum(np.abs(mean_update)) < self.threshold:
        #                    break
        #            elif self.mean_level==1:
        #                self.proposal = self.proposal * getMean(update_hom_vec)
        #            elif self.mean_level==2:
        #                self.proposal=getMean(proposal_vec)
        #            elif self.mean_level==3:
        #                self.proposal = square_to_corners_warp(getMean(pts_vec))
        #if thresh_count==self.n_channels:
        #    break
