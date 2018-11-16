__author__ = 'Tommy'

from SCVUtils import *


class FeatureBase:
    def __init__(self, multi_approach):
        self.multi_approach = multi_approach
        self.n_channels = 1
        self.scv_template = None
        self.scv_intensity_map = None
        if self.multi_approach == 'flatten':
            self.flatten = True
        else:
            self.flatten = False

        if self.multi_approach == 'none' or self.multi_approach == 'flatten':
            self.getSCVExpectation = scv_expectation
            self.getSCVIntensityMap = scv_intensity_map
        else:
            self.getSCVExpectation = scv_expectation_vec
            self.getSCVIntensityMap = scv_intensity_map_vec2

    def initialize(self, n_channels):
        self.n_channels = n_channels

    def getFeature(self, img, pts, warp):
        raise NotImplementedError()

    def getJacobian(self):
        raise NotImplementedError()

    def getSampledRegion(self, img, pts, warp=np.eye(3), result=None, flatten=False):
        raise NotImplementedError()

    def updateSCVIntensityMap(self, img, pts, warp):
        sampled_img = self.getSampledRegion(img, pts, warp, flatten=self.flatten)
        self.scv_intensity_map = self.getSCVIntensityMap(sampled_img, self.scv_template)

    def updateSCVTemplate(self, img, pts, warp):
        self.scv_template = self.getSampledRegion(img, pts, warp, flatten=self.flatten)




