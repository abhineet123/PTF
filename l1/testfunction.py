import numpy
from numpy import matrix as MA
dt = numpy.dtype('f8');

def testfunction(Q):
	(q1,indq) = des_sort(Q);
	print Q
	print q1
	print indq



def des_sort(A):
	B1 = numpy.sort(A);
	Asort = B1[::-1];
	B = numpy.argsort(A) 
	A_ind= B[::-1]
	return Asort, A_ind

if __name__ == '__main__':
	Q = MA([[1e-14],[1e-8],[1e-10],[0.1]]);
	testfunction(Q)
