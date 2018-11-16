import numpy
import math
from numpy import matrix as MA
def aff2image(aff_maps, T_sz):
	dt = numpy.dtype('f8')
	# % height and width of template
	r = T_sz[0,0] # MA(T_sz.shape).item(0);
	c = T_sz[0,1] # MA(T_sz.shape).item(1);
	# number of affine results
	n =  MA(aff_maps.shape).item(1);
	boxes = numpy.zeros((8,n),dt);
	for ii in range(n):
		aff = aff_maps[:,ii];
		R= MA([[aff.item(0), aff.item(1), aff.item(4)],[aff.item(2), aff.item(3), aff.item(5)]]);
		P = MA([[1, r, 1, r], [1, 1, c, c] , [1, 1, 1, 1]]);
		Q = R*P;
		boxes[:,ii]= Q.flatten(1);
	return boxes
