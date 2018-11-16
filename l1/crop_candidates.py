import numpy
from numpy import matrix as MA
#from IMGaffine_r import IMGaffine_r
from IMGaffine import IMGaffine_c
def crop_candidates(img_frame,curr_samples,template_size):
	dt = numpy.dtype('f8')
	nsamples = MA(curr_samples.shape).item(0)
	c = numpy.prod(template_size)
	gly_inrange = MA(numpy.zeros(nsamples),dt)
	gly_crop = MA(numpy.zeros((c,nsamples)),dt)
	for i in range(nsamples):
		curr_afnv = curr_samples[i,:]
		img_cut,gly_inrange[0,i] = IMGaffine_c(img_frame,curr_afnv,template_size)
		gly_crop[:,i] = MA(img_cut).flatten(1).T
	return gly_crop,gly_inrange
