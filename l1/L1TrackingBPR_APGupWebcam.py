import cv2
import numpy
import scipy.ndimage
from numpy import matrix as MA
from InitTemplates import InitTemplates
from APGLASSOup import APGLASSOup
import time
from numpy import linalg
from corners2affine import corners2affine
from affine2image import aff2image
from IMGaffine import IMGaffine_c
import math
from copy import deepcopy
from softresh import softresh_c
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import matplotlib
# from APGLASSOup import APGLASSOup_c
# Give RGB/BGR image as input to the tracking algo
# Else check if RGB or gray and then convert it to gray
# Input the name of the video to be tracked

    #camera =  cv2.VideoCapture(0);
     #while True:
          #f,img = camera.read();
          #cv2.imshow("webcam",img);
          #if (cv2.waitKey (5) != -1):
                #break;



def L1TrackingBPR_APGupWebcam(paraT):
	dt = numpy.dtype('f8')
	framename = 'frameAPG';
	dt2 = numpy.dtype('uint8');
	dt3 = numpy.dtype('float64')
	camera =  cv2.VideoCapture(0);
	f,img = camera.read();
#	import pdb;pdb.set_trace()
	plt.imshow(img)
	pts = [];
	while len(pts) < 4:
	  tellme('Select 4 corners with mouse anticlockwise starting with top left')
	  pts = numpy.asarray( plt.ginput(4,timeout=-1) )
	  if len(pts) < 4:
	    tellme('Too few points, starting over')
	    time.sleep(1) # Wait a second
	plt.close()
	init_pos = MA([[int(pts[0,1]),int(pts[1,1]),int(pts[2,1])],[int(pts[0,0]),int(pts[1,0]),int(pts[2,0])]])
	# paraT is a structure.
	## Initialize templates T
	# Generate T from single image
	init_pos = init_pos;
	n_sample=paraT.n_sample;
	sz_T=paraT.sz_T;
	rel_std_afnv = paraT.rel_std_afnv;
	nT=paraT.nT;
	t = 0;
	# generate the initial templates for the 1st frame. The image has to be in GrayScale
	#Movie = cv2.VideoCapture("Videos/RobotNewSetup.avi")
	#cv2.namedWindow("input")
	#f,img = Movie.read()

	#camera1 =  cv2.VideoCapture(0);
	f,img = camera.read();
#	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
	r1, g1, b1 = img[:,:,0], img[:,:,1], img[:,:,2]
	img = 0.2989*r1 + 0.5870 * g1 + 0.1140 * b1;
	img = MA(img);
	(T,T_norm,T_mean,T_std) = InitTemplates(sz_T,nT,img,init_pos);
	#print T.shape, T_norm.shape,T_mean.shape,T_std.shape
	norms = numpy.multiply(T_norm,T_std); # %template norms
	occlusionNf = 0;
	# L1 function settings
	angle_threshold = paraT.angle_threshold
	#print sz_T.shape
	# import pdb;pdb.set_trace()
	dim_T	= sz_T[0,0]*sz_T[0,1];	# number of elements in one template, sz_T(1)*sz_T(2)=12x15 = 180
	A = MA(numpy.concatenate((T,numpy.matrix(numpy.identity(dim_T))),axis = 1)) # data matrix is composed of T, positive trivial T.
	alpha = 50;# this parameter is used in the calculation of the likelihood of particle filter
	(aff_obj) = corners2affine(init_pos, sz_T); # get affine transformation parameters from the corner points in the first frame
	map_aff = aff_obj['afnv'];
	aff_samples = numpy.dot(numpy.ones((n_sample,1),dt),map_aff);
	T_id	= -numpy.arange(nT);	# % template IDs, for debugging
	fixT = T[:,0]/nT; #  first template is used as a fixed template  CHECK THIS
	# Temaplate Matrix
	Temp = numpy.concatenate((A, fixT), axis=1);
	Dict = numpy.dot(Temp.T,Temp);
	temp1 = numpy.concatenate((T,fixT),axis=1);
	Temp1 = temp1*numpy.linalg.pinv(temp1);
# % Tracking

# % initialization
# nframes = no of frames to be tracked
	nframes = 1340; # pass this as an argument or keyboard interrupt
	temp1 = numpy.concatenate((A,fixT),axis = 1); 
	colDim = MA(temp1.shape).item(1)
	Coeff = numpy.zeros((colDim,nframes),dt);
	count = numpy.zeros((nframes,1),dt);
	
	param = para1(paraT)
	while True:
		start_time= time.time();
		seq = '%05d' % t	
#		filename = '/home/ankush/Desktop/CleanCode/PythonTracker/Videos/Images/cliffbar/imgs/img'+str(seq)+'.png'
		f,img1 = camera.read();
		#		f,img1 = Movie.read() # After some minutes all frames returnes are empty and f is false
		t = t+1;
		print 'Frame number: (%f) \n'% (t);
		r1, g1, b1 = img1[:,:,0], img1[:,:,1], img1[:,:,2]
		img = 0.2989*r1 + 0.5870 * g1 + 0.1140 * b1;
		# Draw transformation samples from a Gaussian distribution
		temp_map_aff = numpy.sum(numpy.multiply(map_aff[0,0:4],map_aff[0,0:4]))/2;
		sc = numpy.sqrt(temp_map_aff);
		std_aff	= numpy.multiply(rel_std_afnv,MA([[1, sc, sc, 1, sc, sc]]));
		map_aff	= map_aff + 1e-14;
		(aff_samples) = draw_sample(aff_samples, std_aff); # draw transformation samples from a Gaussian distribution

		(Y, Y_inrange) = crop_candidates(img, aff_samples[:,0:6], sz_T);

		if numpy.sum(Y_inrange==0) == n_sample:
			print 'Target is out of the frame!\n';
		(Y,Y_crop_mean,Y_crop_std) = whiten(Y);	 # zero-mean-unit-variance
		(Y, Y_crop_norm) = normalizeTemplates(Y); # norm one
		#%-L1-LS for each candidate target
		eta_max	= float("-inf");
		q = numpy.zeros((n_sample,1),dt); #  % minimal error bound initialization
		# % first stage L2-norm bounding
		for j in range(n_sample):
			cond1=Y_inrange[0,j]
			temp_abs = numpy.absolute(Y[:,j])
			cond2 = numpy.sum(temp_abs)
			if cond1 ==0 and cond2==0:
				continue
			# L2 norm bounding
			temp_x_norm = Y[:,j]-Temp1*Y[:,j]
			q[j,0] = numpy.linalg.norm(temp_x_norm);
			q[j,0] = numpy.exp(-alpha*(q[j,0]*q[j,0]));
			
#-------------------------------------------------------------------------------------------------
		# sort samples according to descend order of q
		(qtemp1,indqtemp) = des_sort(q.T);
		q = qtemp1.T;
		indq = indqtemp.T;
    	#second stage
		p= numpy.zeros((n_sample),dt); #observation likelihood initialization
		n = 0;
		tau = 0;
		while (n<n_sample) and (q[n]>=tau):
			APG_arg1 = (Temp.T*Y[:,indq[n]])
			(c) = APGLASSOup(APG_arg1,Dict,param);
#			(c) = APGLASSOup_c(APG_arg1,Dict,param.Lambda, param.Lip, param.Maxit, param.nT)
			c = MA(c);
			Ele1=numpy.concatenate((A[:,0:nT], fixT),axis = 1);
			Ele2=numpy.concatenate((c[0:nT], c[-1]), axis =0 );
			D_s = (Y[:,indq[n-1]] - Ele1*Ele2); #reconstruction error
			D_s = numpy.multiply(D_s,D_s); #reconstruction error
			p[indq[n]] = numpy.exp(-alpha*(numpy.sum(D_s))); #  probability w.r.t samples
			tau = tau + p[indq[n]]/(2*n_sample-1);# update the threshold
			if(numpy.sum(c[1:nT]) < 0): # remove the inverse intensity patterns
				continue;
			elif (p[indq[n]]>eta_max): #******POssilbe Erro*****
				id_max	= indq[n];
				c_max	= c;
				eta_max = p[indq[n]]
			n = n+1;
			count[t-1] = n;
# resample according to probability
		map_aff = aff_samples[id_max,0:6]; # target transformation parameters with the maximum probability
		a_max	= c_max[0:nT,0];
		(aff_samples, _) = resample(aff_samples,p,map_aff); # resample the samples wrt. the probability
		indA = a_max.argmax();
		min_angle = images_angle(Y[:,id_max],A[:,indA]);
     # -Template update
		occlusionNf = occlusionNf-1;
		level = 0.05;
		Initialparameterlambda = MA(param.Lambda);

		if min_angle > angle_threshold and occlusionNf < 0:
			print ('Update!')
			trivial_coef = (numpy.reshape(c_max[nT:-1,0],(sz_T[0,1],sz_T[0,0]))).T;
			dst1 = MA(numpy.zeros((sz_T[0,0],sz_T[0,1])))
			trivial_coef = binary(trivial_coef,level)
			se = MA([[0,0,0,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0],[0,0,0,0,0]]);
			se = numpy.array(se.T)
			areastats,T1 = labels(trivial_coef)
			if T1> 0:
				Area = areastats[:,1];
				max_area = Area.max()
			Area_tolerance= 0.25*sz_T[0,0]*sz_T[0,1];
		# Occlusion Detection
			if T1>0 and max_area < numpy.rint(Area_tolerance):
			# find the template to be replaceed
				tempa_max = a_max[0:nT-1,0];
				indW = tempa_max.argmin();
			# insert new template
				T = MA(T);
				T_id = MA(T_id);
				T_mean = MA(T_mean);
				norms = MA(norms);
				T[:,indW] = Y[:,id_max];
				T_mean[indW,0] = Y_crop_mean[0,indW];
				T_id[0,indW] = t-1; # track the replaced template for debugging
				norms[indW,0] = Y_crop_std[0,id_max]*Y_crop_norm[0,id_max];
			
				(T,_) =  normalizeTemplates(T);
				A[:,0:nT] = T;
			
			# Template Matrix
				Temp = MA(numpy.concatenate((A,fixT),axis=1));
				Dict = Temp.T* Temp;
				tempInverse = numpy.concatenate((T,fixT),axis =1);
				Temp1 = tempInverse*numpy.linalg.pinv(tempInverse);
			else:
				occlusion = 5;
			# update L2 regularized  term
				param.Lambda = MA([[Initialparameterlambda[0,0], Initialparameterlambda[0,1] , 0]]);
				
		elif occlusionNf <0:
			param.Lambda = Initialparameterlambda;
		rect = numpy.rint(aff2image(map_aff.T, sz_T));
		inp	= (numpy.reshape(rect,(4,2))).T;
		
		#import pdb;pdb.set_trace()
		#topleft_r = inp[0,0];
		#topleft_c = inp[1,0];
		#botleft_r = inp[0,1];
		#botleft_c = inp[1,1];
		#topright_r = inp[0,2];
		#topright_c = inp[1,2];
		#botright_r = inp[0,3];
		#botright_c = inp[1,3];
		position = MA([[int(inp[1,0]),int(inp[0,0]),int(inp[1,3]-inp[1,0]),int(inp[0,3]-inp[0,0])]]);
		fdata = open("ResultTracking/L1.txt", "a")
		fdata.write( str(position) +"\n" )      # str() converts to string
		img = numpy.rint(img);
		point1 = (int(inp[1,0]),int(inp[0,0]));
		point2 =  (int(inp[1,2]),int(inp[0,2]));
		point3 = (int(inp[1,3]),int(inp[0,3]));
		point4 = (int(inp[1,1]),int(inp[0,1]));
		print time.time() - start_time
		#if not f:
		#	break
		try:
			cv2.line(img1, point1, point2, (0,0,255), 2)
			cv2.line(img1, point2, point3, (0,0,255), 2)
			cv2.line(img1, point3, point4, (0,0,255), 2)
			cv2.line(img1, point4, point1, (0,0,255), 2)
			cv2.imshow("preview", img1)
#			cv2.imwrite('ResultTracking/{0:05d}.jpg'.format(t),img1)
		except cv2.error as e:
			print e # print error: (-206) Unrecognized or unsupported array type
		k=cv2.waitKey(5)
		if k==27:
			break
	fdata.close()	
 
def des_sort(q):
	B1 = numpy.sort(q);
	Asort = B1[::-1];
	B = numpy.argsort(q) 
	A_ind= B[::-1]
	return Asort, A_ind

def replacement(X,n):
	X = X.flatten(1)
	count = 0;
	for i in range(X.shape[0]):
		if X[i] == n:
			count = count+1;
			X[i] = 0
	return count

def labels(x):
	(X,T) = scipy.ndimage.label(x)
	areastat = MA(numpy.zeros((T,2)))
	for j in range(1,T+1):
		count = replacement(X,j)
		areastat[j-1,:]= [j,count];
	
	return areastat,T

def tellme(s):
    print(s)
    plt.title(s,fontsize=10)
    plt.draw()

def binary(img1, level):
	img1 = MA(img1)
	tempIm = img1.flatten(1)
	for i in range(tempIm.shape[1]):
		if tempIm[0,i] < level:
			tempIm[0,i] = 0;
		else:
			tempIm[0,i] = 1;
	tempIm = (numpy.reshape(tempIm,(img1.shape[1],img1.shape[0]))).T
	return tempIm

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

def normalizeTemplates(A):
	MN = A.shape[0]
	A_norm = numpy.sqrt(numpy.sum(numpy.multiply(A,A),axis=0)) + 1e-14;
	A = numpy.divide(A,(numpy.ones((MN,1))*A_norm));
	return A, A_norm

def resample(curr_samples,prob,afnv):
	dt = numpy.dtype('f8')
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
	
def images_angle(I1,I2):
	I1v = I1.flatten(1)
	col = MA(I1v.shape).item(1);
	I2v = I2.flatten(1)
	I1vn = I1v/(numpy.sqrt(numpy.sum(numpy.multiply(I1v,I1v))) + 1e-14);
	I2vn = I2v/(numpy.sqrt(numpy.sum(numpy.multiply(I2v,I2v)))+ 1e-14);
	angle = math.degrees(math.acos(numpy.sum(numpy.multiply(I1vn,I2vn))))
	return angle

def whiten(In):
	dt = numpy.dtype('f8')
	MN = MA(In.shape).item(0)
	a = numpy.mean(In,axis=0)
	b = numpy.std(In,axis=0) + 1e-14
	out = numpy.divide( ( In - (MA(numpy.ones((MN,1)),dt)*a)), (MA(numpy.ones((MN,1)),dt) *b) )
	return out,a,b
	
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
		
	return x

	
class para1():
		def __init__(self,paraT):
			self.Lambda =  paraT.lambd;
			self.nT = paraT.nT;
			self.Lip = paraT.Lip;
			self.Maxit = paraT.Maxit;


if __name__ == '__main__':
	L1TrackingBPR_APGup(paraT)
