""" 
A collection of functions for applying and generating homographies.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import math
import numpy as np

def homogenize(pts):
    """ Transforms points into their homogeneous coordinate form.

    Parameters:
    -----------
    pts : (n,m) numpy array
      An array where each column represents a point in cartesian
      coordinates.

    Returns:
    --------
    An (n+1, m) numpy array, identical to pts, with a row of ones
    appended to the bottom.

    See Also:
    ---------
    dehomonize
    """
    (h,w) = pts.shape
    results = np.empty((h+1,w))
    results[:h] = pts
    results[-1].fill(1)
    return results

def dehomogenize(pts):
    """ Transforms points into their cartesian coordinate form.
    
    Parameters:
    -----------
    pts : (n,m) numpy array
      An array where each column represents a point in homogeneous
      coordinates. Columns in pts may not be "points at infinity".

    Returns:
    --------
    An (n-1,m) numpy array, where each column is the cartesian
    representation of the corresponding column in the pts matrix.

    See Also:
    ---------
    homogenize
    """
    (h,w) = pts.shape
    results = np.empty((h-1,w))
    results[:h-1] = pts[:h-1]/pts[h-1]
    return results

def apply_to_pts(homography, pts):
    """ Applies a homography to a collection of points.

    Parameters:
    -----------
    homography : (3,3) numpy matrix
      A homography on R^2 represented in homogeneous coordinates.

    pts : (2,n) numpy array
      An array where each column is the cartesian representation
      of a point in R^2. 

    Returns:
    --------
    An (2,n) numpy array, where each column is the image of the
    corresponding column of pts under the givn homography.
    """
    (h,w) = pts.shape    
    result = np.empty((h+1,w))
    result[:h] = pts
    result[-1].fill(1)
    result = np.asmatrix(homography) * result
    result[:h] /= result[-1]
    return np.asarray(result[:h])

def compute_homography(in_pts, out_pts):
    """ Uses the direct linear transform to compute homographies.

    Parameters:
    -----------
    in_pts : (2,n) numpy array
      Each column represents an "input point"
      
    out_pts: (2,n) numpy array
      Each column represents an "output point"

    Returns:
    --------
    A (3,3) numpy matrix H that minimizes:

      l2_norm(apply_to_pts(H, in_pts) - out_pts)^2

    i.e. the homography that does the best job of mapping
    in_pts to out_pts.
    """
    num_pts = in_pts.shape[1]
    in_pts = homogenize(in_pts)
    out_pts = homogenize(out_pts)
    constraint_matrix = np.empty((num_pts*2, 9))
    # print "in_pts=", in_pts
    # print "out_pts=", out_pts
    # print "num_pts=", num_pts
    for i in xrange(num_pts):
        #print "i=", i
        p = in_pts[:,i]
        q = out_pts[:,i]
        constraint_matrix[2*i,:] = np.concatenate([[0,0,0], -p, q[1]*p], axis=0)
        constraint_matrix[2*i+1,:] = np.concatenate([p, [0,0,0], -q[0]*p], axis=0)
    U,S,V = np.linalg.svd(constraint_matrix)
    homography = V[8].reshape((3,3))
    homography /= homography[2,2]
    return np.asmatrix(homography)

_square = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T
def square_to_corners_warp(corners):
    """ Computes the homography from the centered unit square to 
    the quadrilateral given by the corners matrix.
    
    Parameters:
    -----------
    corners : (2,4) numpy array
      The corners of the target quadrilateral.
    
    Returns:
    --------
    A (3,3) numpy matrix, representing a homography, that maps
    the point (-.5,-.5) to corners[:,0], (.5,-.5) to corners[:,1],
    (.5,.5) to corners[:,2], and (-.5,.5) to corners[:,3].

    See Also:
    ---------
    compute_homography
    """
    return compute_homography(_square, corners)

def random_homography(sigma_d, sigma_t):
    """ Generates a random "small" homography.

    For details, please see source.

    Parameters:
    -----------
    sigma_d : real number
      The standard deviation of the noise added to each corner
      of the square.
    
    sigma_t: real number
      The standard deviation of the noise added to the center
      of the square.

    Returns:
    --------
    A (3,3) numpy matrix representing a random homography.
    """
    square = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T
    disturbance = np.random.normal(0,sigma_d,(2,4)) + np.random.normal(0,sigma_t,(2,1))
    return compute_homography(square, square+disturbance)

def random_translation_and_scale(sigma_t, sigma_s):
    """ Generates a random homography that only translates and scales.

    For details, please see source.
    
    Parameters:
    -----------
    sigma_t : real number
      The standard deviation of the translation.

    sigma_s : real number
      The standard deviation of the scale-factor

    See Also:
    ---------
    random_homography
    """
    tx = np.random.normal(0, sigma_t)
    ty = np.random.normal(0, sigma_t)
    s = np.random.normal(1, sigma_s)
    return np.matrix([[1,0,tx],[0,1,ty],[0,0,1/s]])


