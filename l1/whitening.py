import numpy 
from numpy import matrix as MA
def whiten(In):
	dt = numpy.dtype('f8')
	MN = MA(In.shape).item(0)
	a = numpy.mean(In,axis=0)
	b = numpy.std(In,axis=0) + 1e-14
	out = numpy.divide( ( In - (MA(numpy.ones((MN,1)),dt)*a)), (MA(numpy.ones((MN,1)),dt) *b) )
	return out,a,b
	

	
if __name__=='__main__':
	whiten(In)








#function [out,a,b] = whitening(in)
#% whitening an image gallery
#%
#%  in     -- MNxC
#%  out    -- MNxC

#MN = size(in,1);
#a = mean(in);
#b = std(in)+1e-14;
#out = (in - ones(MN,1)*a) ./ (ones(MN,1)*b);
