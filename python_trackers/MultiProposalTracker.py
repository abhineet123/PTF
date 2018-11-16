"""
Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import numpy as np

from ImageUtils import *
from TrackerBase import *

class MultiProposalTracker(TrackerBase):
    
    def __init__(self, tracker, n_proposals, replace_factor, warp_generator, res=None):
        """ Runs a single tracker with multiple proposals.

        Given a single tracker, the MultiProposalTracker will
        maintain several different proposal regions and use the 
        supplied tracker to update each of them. This algorithm 
        assumes that set_region() completely updates the state of 
        the tracking algorithm.

        Parameters:
        -----------
        tracker : TrackerBase
          The tracker used to update each proposal.

        warp_generator : () -> (3,3) numpy array (Homography)
          In order to make sure that the proposals regions don't clump
          together, small random perturbations are applied in some 
          situations. The warp_generator() function should sample
          such a small perturbation warp.

        replace_factor : real
          When the error of a proposal is more than replace_factor * best_error
          we replace it with a random perturbation of the best proposal. This 
          allows us to throw away proposals that have lost track.
        
        See Also:
        ---------
        TrackerBase
        CascadeTracker
        ParallelTracker
        """
        self.tracker = tracker
        self.n_proposals = n_proposals
        self.replace_factor = replace_factor
        self.warp_generator = warp_generator
        self.res = res
        self.proposals = []
        self.best_proposal = None
        self.initialized = False

    def _perturb(self, region):
        old_warp = square_to_corners_warp(region)
        warp =  old_warp * self.warp_generator() * old_warp.I
        return apply_to_pts(warp, region)

    def set_region(self, region):
        self.best_proposal = 0
        self.proposals = [region] + [self._perturb(region) 
                                     for i in xrange(self.n_proposals - 1)]
    
    def sample_region(self, img, region):
        warp = square_to_corners_warp(region)
        warped_pts = apply_to_pts(warp, self.pts)
        return sample_and_normalize(img, warped_pts)
    
    def initialize(self, img, region):
        if self.res == None:
            self.res = _approximate_resolution(region)
        self.pts = res_to_pts(self.res)
        self.tracker.initialize(img, region)
        self.set_region(region)
        self.template = self.sample_region(img, region)
        self.initialized = True

    def update(self, img):
        if not self.initialized: return
        # Update each proposal and compute errors
        errors = np.empty(self.n_proposals)
        for i in xrange(self.n_proposals):
            self.tracker.set_region(self._perturb(self.proposals[i]))
            self.tracker.update(img)
            self.proposals[i] = self.tracker.get_region()
            errors[i] = np.linalg.norm(
                self.sample_region(img, self.tracker.get_region()) - self.template, 1)
        # Compute the best proposal and store its index
        order = np.argsort(errors)
        self.best_proposal = order[0]
        best_error = errors[self.best_proposal]
        # Determine which proposals have too much error
        # and replace them with perturbations of the best proposal.
        for i in xrange(self.n_proposals):
            if errors[i] > self.replace_factor * best_error:
                self.proposals[i] = self._perturb(self.proposals[self.best_proposal])

    def is_initialized(self):
        return self.initialized

    def get_region(self):
        return self.proposals[self.best_proposal]
        
def _approximate_resolution(region):
    def length(i,j):
        return np.linalg.norm(region[:,i] - region[:,j])
    width = length(0,1)/2 + length(2,3)/2
    height = length(1,2)/2 + length(3,0)/2
    return (width, height)
