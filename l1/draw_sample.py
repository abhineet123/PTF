import numpy
from numpy import matrix as MA

def draw_sample(mean_afnv,std_afnv):
	dt = numpy.dtype('f8')
	mean_afnv = MA((mean_afnv),dt);
	nsamples = MA(mean_afnv.shape).item(0)
	MV_LEN=6
	mean_afnv[:,0] = numpy.log(mean_afnv[:,0])
	mean_afnv[:,3] = numpy.log(mean_afnv[:,3])
	outs= MA(numpy.zeros((nsamples,MV_LEN)),dt)
	flatdiagonal = MA((numpy.diagflat(std_afnv)),dt);
	outs[:,0:MV_LEN] = MA(numpy.random.randn(nsamples,MV_LEN),dt) * flatdiagonal + mean_afnv
	outs[:,0] = MA(numpy.exp(outs[:,0]),dt)
	outs[:,3] = MA(numpy.exp(outs[:,3]),dt)
	return outs
