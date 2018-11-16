from numpy import linalg
import numpy as np
from numpy import matrix as MA
from scipy import interpolate
from scipy import weave
from scipy.weave import converters
def affparaminv(est):
    #!!!pay attention
    dt = np.dtype('f8')
    q = MA(np.zeros(2,3),dt)
    q = linalg.pinv(MA([[est[0,2],est[0,3]],[est[0,4],est[0,5]]])) * MA([[-est[0,0],1.0,0.0],[-est[0,1],0,1.0]])
    q=q.flatten(1)
    return MA([[q[0,0],q[0,1],q[0,2],q[0,4],q[0,3],q[0,5]]])

def affparam2geom(est):
    # !!!pay attention
    dt = np.dtype('f8')
    A = MA([[est[0,2],est[0,3]],[est[0,4],est[0,5]]])
    U,S,V = linalg.svd(A,full_matrices=True)
    temp = MA(np.zeros((2,2),dt))
    #temp[0,0] = S[0]
    #temp[1,1] = S[1]
    #S = temp
    #import pdb; pdb.set_trace()
    if(linalg.det(U) < 0):
        U = U[:,range(1,-1,-1)]
        V = V[:,range(1,-1,-1)]
        S = S[:,range(1,-1,-1)]
	temp[1,1] = S[1]
	temp[0,0] = S[0]
	S = temp
    else:
        temp[1,1] = S[0]
        temp[0,0] = S[1]
	S = temp	
    #import pdb; pdb.set_trace()
    q = MA(np.zeros((1,6)),dt)
    q[0,0] = est[0,0]
    q[0,1] = est[0,1]
    q[0,3] = np.arctan2(U[1,0]*V[0,0]+U[1,1]*V[0,1],U[0,0]*V[0,0]+U[0,1]*V[0,1])
    phi = np.arctan2(V[0,1],V[0,0])
    if phi <= -np.pi/2:
        c = np.cos(-np.pi/2)
        s = np.sin(-np.pi/2)
        R = MA([[c,-s],[s,c]])
        V = MA(V) * MA(R)
        S = R.T*MA(S)*R
    if phi > np.pi/2:
        c = np.cos(np.pi/2)
        s = np.sin(np.pi/2)
        R = MA([[c,-s],[s,c]])
        V = MA(V)*MA(R)
        S = R.T*MA(S)*R
    #import pdb; pdb.set_trace()
    q[0,2] = S[0,0]
    q[0,4] = S[1,1]/S[0,0]
    q[0,5] = np.arctan2(V[0,1],V[0,0])
    return q

def warpimg(img,p,sz):
    # arrays
    w = sz[0,0]
    h = sz[0,1]
    result = np.zeros((w*h,p.shape[0]))
    #May have problem    
    x = np.linspace(1-h/2,h/2,h)
    y = np.linspace(1-w/2,w/2,w)
    xv,yv = np.meshgrid(x,y)
    xcoord = np.linspace(1,img.shape[1],img.shape[1])
    ycoord = np.linspace(1,img.shape[0],img.shape[0])
    #f = interpolate.interp2d(xcoord,ycoord,img,kind='cubic')
    #import pdb; pdb.set_trace()
    for i in range(p.shape[0]):
        temp = np.concatenate((np.ones((w*h,1)),xv.reshape((w*h,1),order='F'),yv.reshape((w*h,1),order='F')),axis=1)*MA([[p[i,0],p[i,1]],[p[i,2],p[i,4]],[p[i,3],p[i,5]]])
	#import pdb;pdb.set_trace()
	#tempx = temp[:,0].reshape(h,w).T
	#tempy = temp[:,1].reshape(h,w).T
	#result[:,i] = f(tempx,tempy).reshape(w*h,order='F')
	result[:,i] = sample_region(img, np.array(temp))
    return result

def sample_region(img, temp, result=None):
    """ Samples the image intenisty at a collection of points.

    Notes:
    ------
      - Only works with grayscale images.
      - All points outside the bounds of the image have intensity 128.

    Parameters:
    -----------
    img : (n,m) numpy array
      The image to be sampled from.

    pts : (2,k) numpy array
      The points to be sampled out of the image. These may be sub-pixel
      coordinates, in which case bilinear interpolation is used.

    result : (k) numpy array (optional)
      Optionally you can pass in a results vector which will store the
      sampled vector. If you do not supply one, this function will allocate
      one and return a reference.

    Returns:
    --------
    Returns a (k) numpy array containing the intensities of the given
    sub-pixel coordinates in the provided image.
    """
    num_pts = temp.shape[0]
    (height, width) = img.shape
    if result == None: result = np.empty(num_pts)
    support_code = \
    """
    double bilinear_interp(blitz::Array<double,2> img, int width, int height, double x, double y) {
      using std::floor;
      using std::ceil;
      const int lx = floor(x);
      const int ux = ceil(x);
      const int ly = floor(y);
      const int uy = ceil(y);
      if (lx < 0 || ux >= width || ly < 0 || uy >= height) return 128;
      const double ulv = img(ly,lx);
      const double urv = img(ly,ux);
      const double lrv = img(uy,ux);
      const double llv = img(uy,lx);
      const double dx = x - lx;
      const double dy = y - ly;
      return ulv*(1-dx)*(1-dy) + urv*dx*(1-dy) + llv*(1-dx)*dy + lrv*dx*dy;
    }
    """
    code = \
    """
    int j = 0;
    int k = 1;
    for (int i = 0; i < num_pts; i++) {
      double x = temp(i,j);
      double y = temp(i,k);
      result(i) = bilinear_interp(img, width, height, x, y);
    }
    """
    weave.inline(code, ["img", "result", "temp", "num_pts", "width", "height"],
                 support_code=support_code, headers=["<cmath>"],
                 type_converters=converters.blitz,
                 compiler='gcc')
    return result

def affparam2mat(p):
    # Here I use array instead of matrix
    q = np.zeros(p.shape,'f8')
    p = np.array(p)
    temp = q
    sz = p.shape
    temp1 = p
    #import pdb; pdb.set_trace()
    #if len(p.shape) == 1: 
#	temp = np.zeros(1,p.shape[0])
#	temp[0,:] = p
    s = np.array(p[:,2])
    th = np.array(p[:,3])
    r = np.array(p[:,4])
    phi = np.array(p[:,5])
    cth = np.cos(th)
    sth = np.sin(th)
    cph = np.cos(phi)
    sph = np.sin(phi)
    ccc = cth*cph*cph
    ccs = cth*cph*sph
    css = cth*sph*sph
    scc = sth*cph*cph
    scs = sth*cph*sph
    sss = sth*sph*sph
    q[:,0] = np.array(p[:,0])
    q[:,1] = np.array(p[:,1])
    q[:,2] = s*(ccc +scs +r*(css -scs))
    q[:,3] = s*(r*(ccs -scc) -ccs -sss)
    q[:,4] = s*(scc -ccs +r*(ccs +sss))
    q[:,5] = s*(r*(ccc +scs) -scs +css)
    return MA(q)

def estwarp_condens(img,param_old,n_sample,sz_T,affsig):
        #again array
        #param = np.array(np.tile(affparam2geom(est),[n_sample,1]))
        param = param_old
        #import pdb; pdb.set_trace()
        param = param + np.random.randn(n_sample,6) * np.tile(np.array(affsig),[n_sample,1])
        samples = warpimg(img,affparam2mat(param),sz_T)
        #import pdb; pdb.set_trace()
        return samples,param

def resample2(curr_samples,prob):
        dt = numpy.dtype('f8')
        nsamples = MA(curr_samples.shape).item(0)
        afnv = 0
        #pdb.set_trace()
        if prob.sum() ==0 :
                #import pdb; pdb.set_trace()
                map_afnv = MA(numpy.ones(nsamples),dt).T*afnv
                count = MA(numpy.zeros((prob.shape),dt))
        else:
                prob = prob/prob.sum()
                N = nsamples
                Ninv = 1 / float(N)
                map_afnv = MA(numpy.zeros((N,6)),dt)
                c = pylab.cumsum(prob)
                u = pylab.rand()*Ninv
                i = 0
                #pdb.set_trace()
                for j1 in range(N):
                        uj = u + Ninv*j1
                        while uj > c[i]:
                                i += 1
                        map_afnv[j1,:] = curr_samples[i,:]
                return map_afnv

def weight_eval(samples,template,sz,alpha):
    #wlist = [i for i in range(samples.shape[1])]
    wlist = np.zeros(samples.shape[1])
    samples,_,_  = whiten(samples)
    samples,_ = normalizeTemplates(samples)
    template = template.reshape((sz[0,0]*sz[0,1],1),order='F')
    for i in range(samples.shape[1]):
	temp_norm = samples[:,i] - template
	wlist[i] = np.linalg.norm(temp_norm)
	wlist[i] = np.exp(-alpha*(wlist[i]*wlist[i]))
	if str(wlist[i]) == 'nan':pdb.set_trace()    
    #import pdb; pdb.set_trace()
    return wlist

def drawbox(est,sz):
    temp = np.zeros(6)
    #import pdb;pdb.set_trace()
    temp[:] = est[0,:]
    est = temp
    #temp[:] = est[0,:]
    M = np.array([[est[0],est[2],est[3]],[est[1],est[4],est[5]]])
    w = sz[0,0]
    h = sz[0,1]
    corners = np.array([[1,-w/2,-h/2],[1,w/2,-h/2],[1,w/2,h/2],[1,-w/2,h/2],[1,-w/2,-h/2]]).T
    corners = MA(M) * MA(corners)
    #import pdb; pdb.set_trace()
    return corners

def whiten(In):
    dt = np.dtype('f8')
    MN = MA(In.shape).item(0)
    a = np.mean(In,axis=0)
    b = np.std(In,axis=0) + 1e-14
    out = np.divide( ( In - (MA(np.ones((MN,1)),dt)*a)), (MA(np.ones((MN,1)),dt) *b) )
    return out,a,b

def des_sort(q):
        temp = q
        B1 = numpy.sort(q);
        Asort = B1[::-1];
        B = numpy.argsort(q)
        A_ind= B[::-1]
        #pdb.set_trace()
        return Asort, A_ind

def normalizeTemplates(A):
        MN = A.shape[0]
        A_norm = np.sqrt(np.sum(np.multiply(A,A),axis=0)) + 1e-14;
        A = np.divide(A,(np.ones((MN,1))*A_norm));
        return A, A_norm
