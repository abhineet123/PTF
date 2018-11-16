import numpy
# import scipy
from numpy import matrix as MA
def IMGaffine_r(R_in, AFNV, OSIZE):
	
	# IMGaffine: affine transform the input image and crop the output image
	# with the desired size 
	#   OUT = IMGaffine_r_color(IN, AFNV, OUTSIZE)
	#   OUT:      output image, M x N 
	#   IN:       input image, M_in x N_in
	#   AFNV_OBJ.afnv:     affine parameter [a11, a12, a21, a22, tr, tc]
	#              or transformation matrix [a11 a12 tr; a21 a22 tc; 0, 0, 1];
	#   AFNV_OBJ.size:  output image size
	#
	# The affine parameter a11 is defined as the relative ratio of IN to OUT 
	dt = numpy.dtype('f8');
	
	R = MA([[AFNV.item(0), AFNV.item(1), AFNV.item(4)], [AFNV.item(2), AFNV.item(3), AFNV.item(5)],[0, 0, 1]],dt);
	Rinrow = MA(R_in.shape).item(0); # M_in
	Rincol = MA(R_in.shape).item(1); # N_in
	M = OSIZE.item(0);
	N = OSIZE.item(1);
	# generate 1,2...,M,1,2,..,M
	temp_arrange = (MA(numpy.arange(1,M+1),dt)).T
	temp_ones =MA(numpy.ones((1,N)),dt)
	temp_prod_arr_one = temp_arrange*temp_ones
	P = MA(numpy.empty((3,M*N)),dt)
	P[0,:] = temp_prod_arr_one.flatten(1) # (numpy.reshape(temp_prod_arr_one,(1,M*N))) 
	#  generate 1,1,..,1,2,2,..2,...,N,N...N
	temp_arrange = MA(numpy.arange(1,N+1),dt)
	temp_ones = MA(numpy.ones((M,1)),dt)
	tempP1 = temp_ones*temp_arrange
	P[1,:] = tempP1.flatten(1);
	P[2,:] = MA(numpy.ones((1, M*N)),dt)
	K= R*P;
	
	#  Q = round(R*P);
	Q = numpy.rint(K);
	R_out1 = MA(numpy.zeros((M*N, 1)));
		
	condition = (Q[0,:] >= 1) & (Q[0,:] <= Rinrow) & (Q[1,:] >=1) & (Q[1,:]<Rincol) 
	jj = numpy.nonzero(condition) # %find the index in the first and second row satisfy such condition
	#import pdb; pdb.set_trace()
	#if len(j):
	#	j = numpy.transpose(j)
	# import pdb; pdb.set_trace()
	# j = MA(j)
	j = jj[1];
	j = j.T;
	
	if j.size>0: # % number of indices bigger than 0
		in_range = 1;
		temp_slice = (Q[1,j]-1)*Rinrow + Q[0,j]
		temp_slice = MA(temp_slice,int)
		
		R_in_temp = numpy.ravel(R_in,order='F')
		
		R_in_temp = MA(R_in_temp)
		R_in_temp = R_in_temp.T
		R_out1[j,0] = R_in_temp[temp_slice,0];#% set the values to the colume****ERROR POSSIBLE ****
		a = numpy.mean(R_out1[j]);
		temp_arrange = MA(numpy.arange(0,M*N),dt)
		temp_arrange = numpy.array(temp_arrange)
		j = numpy.array(j)
		bb =numpy.setdiff1d(temp_arrange,j)
		# import pdb; pdb.set_trace()
		if bb.size > 0: 
			bb = MA(bb,int) - numpy.ones((bb.size,1))
			bb = MA(bb,int)
			R_out1[bb,0] = a; # % set other index equal to the mean
		R_out = numpy.reshape(R_out1, (N,M));
		R_out = MA(R_out,dt).T

	else:
		in_range = 0;
		R_out = R_out1;

	return R_out, in_range
	
if __name__ == '__main__':
	IMGaffine_r(AFNV)
