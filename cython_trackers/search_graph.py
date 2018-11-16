import random
import math
import numpy
from numpy import matrix as MA
from knnsearch import knnsearch
from build_graph import build_graph
import pdb

def  search_graph(query, nodes, DS, K):

    ''' Authors: Ankush Roy (ankush2@ualberta.ca)
                 Kiana Hajebi (hajebi@ualberta.ca)
    '''
    dt = numpy.dtype('f8')
    nodes = MA(nodes,dt)
    query = MA(query,dt)
    DS = MA(DS, dt)
    random.seed(100)
    k = nodes.shape[1]
    depth = 0
    flag  = 0
    parent_id = randd(1, nodes.shape[0], 1); # Check this
    visited = 1;
    while 1:
        parent_vec = MA(DS[int(parent_id),0:],dt)  # parent node
        parent_dist = numpy.sqrt((query - parent_vec) * numpy.transpose(query - parent_vec))
        child_ids = nodes[int(parent_id),0:];
        Val = numpy.zeros((child_ids.shape[1],DS.shape[1]),dt)
        I = 0 
        while I < child_ids.shape[1]:
            Val[I,0:] = DS[int(child_ids[0,I]),0:]
            I += 1
        Val = MA(Val)
        (nn1_ind, nn1_dist) = knnsearch(query,Val, K)
        visited = visited + k
        if (parent_dist <= nn1_dist):
            flag=1
            break
        parent_id = child_ids[0,nn1_ind]
        depth = depth+1
    if flag == 1:
        nn_id = parent_id
        nn_dist  = parent_dist
    else:
        nn_id = - 1
        nn_dist = -1
    return nn_id, nn_dist, visited


def randd(lb, up, number):
    ind = []
    up = up + 1
    if number >= up:
        print('number should be <= up + 1')
        return
    a,b = lb, up
    n = 1
    ind = math.floor(a + (b-a)*random.random())
    while n < number:
        r = floor(a + (b-a)*random.randd())
        if isempty(find(ind == r)):
            ind = numpy.hstack([ind, r])
            n = n + 1
    return ind

    

