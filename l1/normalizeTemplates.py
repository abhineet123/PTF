import numpy

def normalizeTemplates(A):
	MN = A.shape[0]
	A_norm = numpy.sqrt(numpy.sum(numpy.multiply(A,A),axis=0)) + 1e-14;
	A = numpy.divide(A,(numpy.ones((MN,1))*A_norm));
	return A, A_norm
