__author__ = 'Tommy'
from FeatureBase import *
from ImageUtils import *


class Histogram(FeatureBase):
    def __init__(self, min_bin=0, max_bin=255, bin_count=256, val_type='int',
                 multi_approach='none', normalize=True):

        FeatureBase.__init__(self, multi_approach)

        if self.multi_approach == 'none':
            self.getSampledRegion = sample_region
        else:
            self.getSampledRegion = sample_region_vec

        if self.multi_approach == 'none' or self.multi_approach == 'flatten':
            self.getFeature = self.getHistogramOfColors
            self.getJacobian = self.getJacobianOfHistogram
        else:
            self.getFeature = self.getHistogramOfColorsVec
            self.getJacobian = self.getJacobianOfHistogramVec

        self.min_bin = min_bin
        self.max_bin = max_bin
        self.bin_count = bin_count
        self.normalize = normalize
        self.val_type = val_type
        self.hist=None

    def getHistogramOfColors(self, img, pts, warp, use_scv=False):
        # print 'Starting getHistogramOfColors'
        sampled_img = self.getSampledRegion(img, pts, warp, flatten=self.flatten)

        if use_scv and self.scv_intensity_map is not None:
            sampled_img = self.getSCVExpectation(sampled_img, self.scv_intensity_map)

        n_pts = sampled_img.shape[0]
        hist = np.zeros(256, dtype=np.float64)
        code = \
            """
            for (int i = 0; i < n_pts; i++) {
                int intensity=sampled_img(i);
                hist(intensity)++;
            }
            """
        # print 'running weave in getHistogram'
        weave.inline(code, ["sampled_img", "hist", "n_pts"],
                     type_converters=converters.blitz,
                     compiler='gcc')
        # print 'done'
        # np.savetxt('hist_orig.txt', hist.T, fmt='%12.6f', delimiter='\t')
        if self.normalize:
            hist /= n_pts
        # np.savetxt('hist_norm.txt', hist.T, fmt='%12.6f', delimiter='\t')
        # np.savetxt('img.txt', img.T, fmt='%f')
        # np.savetxt('hist.txt', hist.T, fmt='%f')
        # print 'done function'
        return hist

    def getHistogramOfColorsVec(self, img, pts, warp, use_scv=False):
        # print 'Starting getHistogramOfColorsVec'
        sampled_img = self.getSampledRegion(img, pts, warp, flatten=self.flatten)
        if use_scv and self.scv_intensity_map is not None:
            sampled_img = self.getSCVExpectation(sampled_img, self.scv_intensity_map)

        n_channels, n_pts = sampled_img.shape
        hist = np.zeros((n_channels, 256), dtype=np.float64)
        # print 'n_channels=', n_channels
        # print 'n_pts=', n_pts
        code = \
            """
            for (int i = 0; i < n_channels; i++) {
                for (int j = 0; j < n_pts; j++) {
                    int intensity=sampled_img(i, j);
                    hist(i, intensity)++;
                }
            }
            """
        # print 'running weave in getHistogramVec'
        weave.inline(code, ["sampled_img", "hist", "n_pts", "n_channels"],
                     type_converters=converters.blitz,
                     compiler='gcc')
        # print 'done'
        # np.savetxt('hist_orig.txt', hist.T, fmt='%12.6f', delimiter='\t')
        if self.normalize:
            hist /= n_pts
        # np.savetxt('hist_norm.txt', hist.T, fmt='%12.6f', delimiter='\t')
        # np.savetxt('img_mean.txt', img.T, fmt='%f')
        # np.savetxt('hist_mean.txt', hist.T, fmt='%f')
        # print 'done function'
        self.hist=hist.copy()
        return hist

    def getJacobianOfHistogram(self, img, pts, initial_warp, eps=1e-10):
        # print 'Starting getJacobianOfHistogram'

        def f(p):
            W = initial_warp * make_hom_sl3(p)
            return self.getFeature(img, pts, W)

        jacobian = np.empty((self.bin_count, 8))
        for i in xrange(0, 8):
            o = np.zeros(8)
            o[i] = eps
            jacobian[:, i] = (f(o) - f(-o)) / (2 * eps)
        # print 'done function'
        if len(np.nonzero(jacobian)) == 0:
            raise SystemExit('Jacobian is zero')

        return np.asmatrix(jacobian)

    def getJacobianOfHistogramVec(self, img, pts, initial_warp, eps=1e-10):
        # print 'Starting getJacobianOfHistogramVec'

        def f(p):
            W = initial_warp * make_hom_sl3(p)
            return self.getFeature(img, pts, W)

        jacobian = np.empty((self.n_channels, self.bin_count, 8))
        for j in xrange(0, 8):
            o = np.zeros(8)
            o[j] = eps
            jacobian[:, :, j] = (f(o) - f(-o)) / (2 * eps)

        if len(np.nonzero(jacobian)) == 0:
            raise SystemExit('Jacobian is zero')
        before_file = open('jacobian_before.txt', 'a')
        for i in xrange(self.n_channels):
            before_file.write('channel %d\n' % i)
            np.savetxt(before_file, jacobian[i, :, :], fmt='%12.6f', delimiter='\t')
        before_file.close()

        # jacobian.reshape(self.n_channels, self.bin_count, 8)
        # after_file=open('jacobian_after.txt', 'a')
        # for i in xrange(self.n_channels):
        # after_file.write('channel %d\n'%i)
        # np.savetxt(after_file, jacobian[i, :, :])
        # after_file.close()

        # if jacobian is None:
        # jacobian=jacobian_ch.copy()
        # else:
        # jacobian=np.dstack((jacobian, jacobian_ch))

        # print 'done function'
        return jacobian
