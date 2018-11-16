import random
import math
import numpy 
from numpy import matrix as MA
def randd(lb, up, number):

    ''' Authors: Ankush Roy (ankush2@ualberta.ca)
                 Kiana Hajebi (hajebi@ualberta.ca)
    '''
    
    ind = []
    up=up+1
    if number>=up:
        print('number should be <= up+1')
        return

    a,b=lb,up
    n=1
    ind = math.floor(a + (b-a)* random.random())
    while n < number:
        r = floor(a + (b-a)*random.random())
        print ind
        if  isempty(find(ind==r)):  #any(ind==r):
            ind = numpy.hstack([ind,r])
            n=n+1
    return ind

