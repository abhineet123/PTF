import numpy as np
import pyflann
from ImageUtils import *
import itertools
import operator


class WarpIndex:
    """ Utility class for building and querying the set of reference images/warps. """

    def __init__(self, n_samples, warp_generator, img, pts, initial_warp, res,
                 feature_obj):
        self.resx = res[0]
        self.resy = res[1]
        self.sift = False
        self.indx = []
        n_points = pts.shape[1]
        print "Sampling Warps..."
        self.warps = [np.asmatrix(np.eye(3))] + [warp_generator() for i in xrange(n_samples - 1)]
        print "Sampling Images..."
        self.images = None
        self.getBestMatch = self.best_match
        self.feature_obj = feature_obj
        for i, w in enumerate(self.warps):
            sample = self.feature_obj.getFeature(img, pts, initial_warp * w.I)
            if self.images is None:
                self.images = np.empty(sample.shape + (n_samples, ))
            self.images[:, i] = sample
            # self.images[:,i] = sample_and_normalize(img, apply_to_pts(initial_warp * w.I, pts))
        print "Building FLANN Index..."
        # pyflann.set_distance_type("manhattan")
        if self.sift == False:
            self.flann = pyflann.FLANN()
            # print(self.images.shape)
            self.flann.build_index(self.images.T, algorithm='kdtree', trees=10)
        else:
            desc = self.list2array(self.pixel2sift(self.images))
            # --- Building Flann Index --- #
            self.flann = pyflann.FLANN()
            # self.flann.build_index(np.asarray(self.images).T, algorithm='linear')
            #print(type(desc))
            #pdb.set_trace()
            self.flann.build_index(desc.T, algorithm='kdtree', trees=10)
        print "Done!"

    # --- For sift --- #
    def pixel2sift(self, images):
        detector = cv2.FeatureDetector_create("SIFT")
        detector.setDouble('edgeThreshold', 30)
        descriptor = cv2.DescriptorExtractor_create("SIFT")
        # sift = cv2.SIFT(edgeThreshold = 20)
        # -- store descriptors in list --#
        desc = []
        for i in range(images.shape[1]):
            patch = (images[:, i].reshape(self.resx, self.resy)).astype(np.uint8)
            # pdb.set_trace()
            skp = detector.detect(patch)
            skp, sd = descriptor.compute(patch, skp)
            desc.append(sd)
            self.indx.append(len(skp))
        return desc

    # --- For sift ---#
    def list2array(self, desc):
        nums = sum(self.indx)
        descs = np.empty((128, nums), dtype=np.float64)
        counts = 0
        for item in desc:
            if item == None:
                continue
            for j in range(item.shape[0]):
                descs[:, counts] = item[j, :].T
                counts += 1
        return descs.astype(np.float32)

    # ---SIFT function --- #
    def best_match_sift(self, desc):
        # print(type(desc))
        results, dists = self.flann.nn_index(desc)
        index = int(results[0])
        index += 1
        count = 0
        for item in self.indx:
            if index <= item:
                result = count
            else:
                index -= item
                count += 1
        return self.warps[count], dists[0], count

    def best_match(self, img):
        # print(img.shape)
        results, dists = self.flann.nn_index(img)
        return self.warps[results[0]]


class WarpIndexVec:
    """ Utility class for building and querying the set of reference images/warps. """

    def __init__(self, n_samples, warp_generator, img, pts, n_channels, initial_warp, res,
                 multi_approach, feature_obj):
        if len(img.shape) < 3:
            raise AssertionError("Error in _WarpIndexVec: The image is not multi channel")

        self.n_channels = n_channels
        self.resx = res[0]
        self.resy = res[1]
        self.sift = False
        self.flatten = False
        self.use_mean = False
        self.flann_vec = None
        self.flann = None

        if multi_approach == 'flatten':
            self.flatten = True
        elif multi_approach == 'mean':
            self.use_mean = True
        elif multi_approach == 'majority':
            pass
        else:
            raise SystemExit('Error in _WarpIndexVec:'
                             'Invalid multi approach ', multi_approach)

        self.images = None
        self.getBestMatch = None
        self.feature_obj = feature_obj

        print "Sampling Warps..."
        self.warps = [np.asmatrix(np.eye(3))] + [warp_generator() for i in xrange(n_samples - 1)]
        # print 'self.warps=\n', self.warps
        # raw_input("Press Enter to continue...")

        print "Sampling Images..."
        for i, w in enumerate(self.warps):
            sample = self.feature_obj.getFeature(img, pts, initial_warp * w.I)

            if self.images is None:
                self.images = np.empty(sample.shape + (n_samples, ))

            if self.flatten:
                self.images[:, i] = sample
            else:
                self.images[:, :, i] = sample

        print "Building FLANN Index..."
        # pyflann.set_distance_type("manhattan")
        if self.flatten:
            self.flann = pyflann.FLANN()
            # print 'self.images.shape:',  self.images.shape
            #print 'self.images.dtype',self.images.dtype
            self.flann.build_index(self.images.T, algorithm='kdtree', trees=10)
            self.getBestMatch = self.best_match
        else:
            self.flann_vec = []
            for i in xrange(self.n_channels):
                current_images = self.images[i, :, :]
                ch_flann = pyflann.FLANN()
                # print(self.images.shape)
                ch_flann.build_index(current_images.T, algorithm='kdtree', trees=10)
                self.flann_vec.append(ch_flann)
            self.getBestMatch = self.best_match_vec
        print "Done!"

    def best_match(self, img):
        # print 'img.shape',img.shape
        # print 'img.dtype',img.dtype
        results, dists = self.flann.nn_index(img)
        return self.warps[results[0]]

    def best_match_vec(self, img):
        warp_sum = None
        result_vec = []
        # print '\n\n'
        for i in xrange(self.n_channels):
            results, dists = self.flann_vec[i].nn_index(img[i, :])
            result_vec.append(results[0])
            current_warp = self.warps[results[0]].copy()

            # print 'results=\n', results
            # print 'self.warps=\n', self.warps
            # print 'current_warp=\n', current_warp
            if warp_sum is None:
                warp_sum = current_warp
            else:
                warp_sum += current_warp
        # print '\n\n'
        if self.use_mean:
            warp = warp_sum / self.n_channels
            # print 'warp_sum=\n', warp_sum
            # print 'warp=\n', warp
        else:
            # print "result_vec=", result_vec
            most_common_res = most_common(result_vec)
            if result_vec.count(most_common_res) >= len(result_vec) / 2:
                # print "most_common_res=", most_common_res
                warp = self.warps[most_common_res]
            else:
                warp = warp_sum / self.n_channels
        return warp


def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
            # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

        # pick the highest-count/earliest item

    return max(groups, key=_auxfun)[0]
