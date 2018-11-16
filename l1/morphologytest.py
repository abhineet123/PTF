import numpy
import scipy.ndimage
from numpy import matrix as MA


def relacement(X,n):
	X = X.flatten(1)
	count = 0;
	for i in range(X.shape[0]):
		if X[i] == n:
			count = count+1;
			X[i] = 0
	return count

def labels(x):
	(X,T) = scipy.ndimage.label(x)
	print X
	areastat = MA(numpy.zeros((T,2)))
	for j in range(1,T+1):
		# import pdb;pdb.set_trace()
		count = relacement(X,j)
		# import pdb;pdb.set_trace()
		areastat[j-1,:]= [j,count];
	
	return areastat
	
if __name__ == '__main__':
	x= MA([[0,0,1,0,0,1],[0,0,1,1,0,1],[0,0,0,0,0,0],[0,0,1,0,1,0],[0,1,1,1,1,0]])
	labels(x)
