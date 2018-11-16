__author__ = 'Tommy'
from FeatureBase import *
from ImageUtils import *
from SCVUtils import *

class NoFeature(FeatureBase):
    def __init__(self, multi_approach='none'):

        FeatureBase.__init__(self, multi_approach)

        if self.multi_approach == 'none':
            self.getSampledRegion = sample_region
        else:
            self.getSampledRegion = sample_region_vec

        if self.multi_approach == 'none' or self.multi_approach == 'flatten':
            self.getJacobian = self.getJacobianOfImage
        else:
            self.getJacobian = self.getJacobianOfImageVec

    def getFeature(self, img, pts, warp, use_scv=False):
        sampled_img = self.getSampledRegion(img, pts, warp, flatten=self.flatten)
        if use_scv and self.scv_intensity_map is not None:
            sampled_img = self.getSCVExpectation(sampled_img, self.scv_intensity_map)
        return sampled_img

    def getJacobianOfImage(self, img, pts, initial_warp, eps=1e-10):
        n_pts = pts.shape[1] * self.n_channels

        def f(p):
            W = initial_warp * make_hom_sl3(p)
            return self.getFeature(img, pts, W)

        jacobian = np.empty((n_pts, 8))
        for i in xrange(0, 8):
            o = np.zeros(8)
            o[i] = eps
            jacobian[:, i] = (f(o) - f(-o)) / (2 * eps)
        return np.asmatrix(jacobian)

    def getJacobianOfImageVec(self, img, pts, initial_warp, eps=1e-10):
        n_pts = pts.shape[1]

        def f(p):
            W = initial_warp * make_hom_sl3(p)
            return self.getFeature(img, pts, W)

        jacobian = np.empty((self.n_channels, n_pts, 8))
        for i in xrange(0, 8):
            o = np.zeros(8)
            o[i] = eps
            jacobian[:, :, i] = (f(o) - f(-o)) / (2 * eps)
        return jacobian