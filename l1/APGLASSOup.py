import numpy
from numpy import matrix as MA
from copy import deepcopy
from softresh import softresh_c

def APGLASSOup(b,A,para):
	dt = numpy.dtype('f8')
	colDim = MA(A.shape).item(0)
	xPrev = MA(numpy.zeros((colDim,1)),dt)
	x= MA(numpy.zeros((colDim,1)),dt)
	tPrev =1
	t=1
	lambd = MA(para.Lambda)
	Lip = para.Lip
	maxit = para.Maxit
	nT = para.nT
	temp_lambda = MA(numpy.zeros((colDim,1)),dt)
	temp_lambda[0:nT,0] = lambd[0,0]
	temp_lambda[:,-1][0,0] = lambd[0,0]
	for i in range(maxit):
		temp_t = (tPrev-1)/t
		temp_y = (1+temp_t)*x - temp_t *xPrev;
		temp_lambda[nT:-1,0] = lambd[0,2]*temp_y[nT:-1,0]
		temp_y = temp_y - (A*temp_y-b + temp_lambda)/Lip
		xPrev=deepcopy(x) # Python by default copies by reference so changed to copy by value
		for j in range(nT):
			x[j,0] = max(temp_y[j,0].max(),0)
		x[-1,0] = max(temp_y[-1,0],0)
		y_input = temp_y[nT:-1,0]
		x[nT:-1,0] = softresh_c(y_input,lambd[0,1]/Lip)
		tPrev = t
		t = (1+numpy.sqrt(1+4*t*t))/2
	c=x
	return c

#def  softresh(x,lam):
	#dt = numpy.dtype('f8');
	#y = MA(numpy.zeros((x.shape[0],x.shape[1]),dt))
	#for i in range(x.shape[0]):
		#y[i,0] = max((x[i,0]-lam),0)-max((-x[i,0]-lam),0)
	#return y
	
