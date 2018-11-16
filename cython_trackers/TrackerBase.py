"""
The ahstract base class for all tracking algorithms. 
Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import numpy as np
from utility import rectangle_to_region

class TrackerBase:
    """ The base class for all tracking algorithms.

    This class serves two purposes. First, it demonstrates
    the minimal interface expected by a tracking algorithm.
    Second, it implements various algorithm-independent 
    converstions - mainly convenience functions for working
    with rectangles instead of arbitrary quadrilaterals.
    """

    def set_region(self, corners):
        """ Sets the tracker's current state.
        
        Parameters:
        -----------
        corners : (2,4) numpy array
          An array where each column is one corner of the target region.
          They should come in the following order:
            corners[:,0] = template upper left corner
            corners[:,1] = template upper right corner
            corners[:,2] = template lower right corner
            corners[:,3] = template lower left corner
        
        See Also:
        ---------
        set_region_with_rectangle
        """
        raise NotImplementedError()

    def set_region_with_rectangle(self, ul, lr): 
        """ Sets the tracker's current state.
        
        Parameters:
        -----------
        ul : (real, real)
          A tuple representing the pixel coordinates of the upper left
          corner of the target region
        lr : (real, real)
          A tuple representing the pixel coordinates of the lower right
          corner of the target region
        
        See Also:
        ---------
        set_region
        """
        self.set_region(rectangle_to_region(ul,lr))

    def get_region(self): 
        """ Returns the four corners of the target region. See set_region
        for more information on the format.

        See Also:
        ---------
        set_region
        """
        raise NotImplementedError()

    def get_warp(self):
        """ Gets the tracker's current warp.
        
        See Also:
        ---------
        set_warp
        """
        raise NotImplementedError()

    def set_warp(self, warp):
        """ Sets the tracker's current warp.

        See Also:
        ---------
        get_warp
        """
        raise NotImplementedError()

    def initialize(self, img, region):
        """ Initializes the tracker.

        This function indicates to the tracking algorithm what
        the target object is.

        Parameters:
        -----------
        img : (n,m) numpy array
          The frame containing the image of the target object.
        
        region : (2,4) numpy array
          The corners of the target region. See set_region for details
        
        See Also:
        ---------
        initialize_with_rectangle
        """
        raise NotImplementedError()

    def initialize_with_rectangle(self, img, ul, lr):
        """ Initializes the tracker.
        
        Same as initialize except the target region is specified
        using the upper left and lower right corners only.
        
        See Also:
        ---------
        initialize
        """
        self.initialize(img, rectangle_to_region(ul,lr))
    
    def update(self, img): 
        """ Updates the tracker state. 
        
        This function should be called once for each frame in the video.

        Parameters:
        -----------
        img : (n,m) numpy array
          The most recent image in the video sequence. 
        """
        raise NotImplementedError()

    def is_initialized(self): 
        """ Returns whether the tracker is initialized yet or not. """
        raise NotImplementedError()
