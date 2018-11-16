from knnsearch import knnsearch
import random
import numpy
from numpy import matrix as MA
def  build_graph(X, k):

    ''' Build a Connected graph using k neighbours
        Author: Ankush Roy (ankush2@ualberta.ca)
                Kiana Hajebi (hajebi@ualberta.ca)
    '''
    dt = numpy.dtype('f8')
    f=[]
    nodes = numpy.zeros((X.shape[0],k),dt)
    print k 
    for i in range(X.shape[0]):
        #print 'building %d'%i
        query = MA(X[(i-1),0:],dt)
        (nns_inds, nns_dists) = knnsearch(query,X,k+1)
        I  = 0 
        f = []
        while I < len(nns_inds):
            if nns_inds[I] == i-1:
                nns_inds.remove(i-1)
                nodes[i-1,0:] = nns_inds
                break            
            else:
                I += 1
    return nodes


