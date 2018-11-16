import math
import numpy
from numpy import matrix as MA
from copy import deepcopy
def knnsearch(Q,X,k):

    ''' Authors: Ankush Roy (ankush2@ualberta.ca)
                 Kiana Hajebi (hajebi@ualberta.ca)
    '''
    
    index1 = []
    dist = []
    for i in range(len(X)): 
        distVec = (X[i,0:] - Q)*numpy.transpose(X[i,0:] - Q)
        dist.append(math.sqrt(sum(distVec)))
    Dist = deepcopy(dist)
    dist.sort()
    minDist = dist[0:k]
    I = 0 
    while I < k:
        index1.append(Dist.index(dist[I]))
        I += 1
    return  index1, minDist
                  
        


