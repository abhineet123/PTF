""" 
Paramaterization of the set of homography using the lie algebra sl(3)
associated to the special linear group SL(3). This has the advantage of
only producing homographie matrices with det(H) = 1.

For details, see

S. Benhimane and E. Malis, "Real-time image-based tracking of planes
using efficient second-order minimization," Intelligent Robots and Systems, 2004.
(IROS 2004). Proceedings. 2004 IEEE/RSJ International Conference on, vol. 1, 
pp. 943-948 vol. 1, 2004.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import numpy as np
from scipy.linalg import expm

_sl3_basis = map(lambda x: np.matrix(x, dtype=np.float64), [
        [[0,1,0],
         [0,0,0],
         [0,0,0]],
        [[0,0,1],
         [0,0,0],
         [0,0,0]],
        [[0,0,0],
         [0,0,1],
         [0,0,0]],
        [[0,0,0],
         [1,0,0],
         [0,0,0]],
        [[0,0,0],
         [0,0,0],
         [1,0,0]],
        [[0,0,0],
         [0,0,0],
         [0,1,0]],
        [[1,0,0],
         [0,-1,0],
         [0,0,0]],
        [[0,0,0],
         [0,1,0],
         [0,0,-1]]]
                 )

def make_hom_sl3(p):
    log = 0
    for i in xrange(8):
        log += p[i] * _sl3_basis[i]
    return np.asmatrix(expm(log))
