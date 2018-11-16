"""
Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import numpy as np

from ImageUtils import *
from TrackerBase import *

class ParallelTracker(TrackerBase):
    
    def __init__(self, trackers):
        """ Allows multiple trackers to be combined in parallel
        
        Given a collection of tracking algorithms, the ParallelTracker
        will compute the update proposed by each of the trackers and
        select the one with the lowest discrepancy between the template
        and the proposed region. 

        Parameters:
        -----------
        trackers : [TrackerBase]
          trackers is a list of objects, each implementing the TrackerBase
          interface.

        See Also:
        ---------
        TrackerBase
        CascadeTracker
        MultiProposalTracker
        """
        self.trackers = trackers
        self.initialized = False

    def set_region(self, region):
        for t in self.trackers: 
            t.set_region(region)
        self.region = region

    def sample_region(self, img, region):
        warp = square_to_corners_warp(region)
        warped_pts = apply_to_pts(warp, self.pts)
        return sample_and_normalize(img, warped_pts)

    def initialize(self, img, region):
        self.res = _approximate_resolution(region)
        self.pts = res_to_pts(self.res)
        for t in self.trackers: 
            t.initialize(img, region)
        self.region = region
        self.template = self.sample_region(img, self.region)
        self.initialized = True

    def update(self, img):
        if not self.initialized: return
        best_error = float("inf")
        for t in self.trackers:
            t.set_region(self.region)
            t.update(img)
            error = np.linalg.norm(self.sample_region(img, t.get_region()) - self.template,1)
            if error < best_error:
                best_error = error
                self.region = t.get_region()

    def is_initialized(self):
        return self.initialized

    def get_region(self):
        return self.region

def _approximate_resolution(region):
    def length(i,j): return np.linalg.norm(region[:,i] - region[:,j])
    width = length(0,1)/2 + length(2,3)/2
    height = length(1,2)/2 + length(3,0)/2
    return (width, height)

