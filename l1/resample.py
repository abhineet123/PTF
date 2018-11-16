import numpy
from numpy import matrix as MA
import math

def resample(curr_samples,prob,afnv):
	dt = numpy.dtype('f8');
	nsamples = MA(curr_samples.shape).item(0)
	if prob.sum() ==0 :
		map_afnv = MA(numpy.ones(nsamples),dt)*afnv
		count = MA(numpy.zeros((prob.shape),dt))
	else:
		prob = prob/(prob.sum())
		count = MA(numpy.ceil(nsamples*prob),int)
		count = count.T
		map_afnv = MA(numpy.zeros((1,6)),dt);
	
		for i in range(nsamples):
			for j in range(count[i]):
				map_afnv = MA(numpy.concatenate((map_afnv,curr_samples[i,:]),axis=0),dt)
		
		K = map_afnv.shape[0];
		map_afnv = map_afnv[1:K,:];
		ns = count.sum()
		if nsamples > ns:
			map_afnv = MA(numpy.concatenate((map_afnv,MA(numpy.ones(((nsamples-ns),1))*afnv,dt)),axis=0),dt)
		
		map_afnv = map_afnv[0:nsamples,:]
	
	return map_afnv,count
	
	
	
