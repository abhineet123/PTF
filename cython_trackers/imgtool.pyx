from numpy import linalg
import numpy
import numpy as np
from numpy import matrix as MA
import pdb
import pylab
#from scipy import interpolate
#from scipy import weave
#from scipy.weave import converters
from utility cimport scv_intensity_map,scv_expected_img
import cython
from cython.parallel cimport prange
import time

cdef extern from "math.h": 
    double floor(double)  
    double ceil(double) 
    double sqrt(double)

# compute affine parameters that when used to warp the
# normalized corners will give the given corners where
#  the normalized corners goo from -res/2 to res/2
# and are thus centered at origin
def affinv(corners, sz):
    w = sz[0,0]
    h = sz[0,1]
    A = np.ones((4,3), dtype=np.float64)
    C = np.ones((4,2), dtype=np.float64)
    A[0,1] = -w/2.0
    A[0,2] = -h/2.0
    A[1,1] = w/2.0
    A[1,2] = -h/2.0
    A[2,1] = w/2.0
    A[2,2] = h/2.0
    A[3,1] = -w/2.0
    A[3,2] = h/2.0
    C[:,0] = corners[0,:].T
    C[:,1] = corners[1,:].T
    #res = linalg.pinv(C) * MA(A)
    res = linalg.lstsq(A, C)[0]
    res1 = np.empty((1,6), dtype=np.float64)
    res1[0,0] = res[0,0]
    res1[0,1] = res[0,1]
    res1[0,2] = res[1,0]
    res1[0,3] = res[2,0]
    res1[0,4] = res[1,1]
    res1[0,5] = res[2,1]
    #pdb.set_trace()
    return res1 

# converts affine matrix entries to some kind of weird
# geometric coordinates that are apparently easier to sample from;
def affparam2geom(est):
    # !!!pay attention
    dt = np.dtype('f8')
    A = MA([[est[0,2],est[0,3]],[est[0,4],est[0,5]]])
    U,S,V = linalg.svd(A,full_matrices=True)
    V = V.T
    temp = MA(np.zeros((2,2),dt))
    temp[0,0] = S[0]
    temp[1,1] = S[1]
    S = temp
    #S = temp
    #import pdb; pdb.set_trace()
#    if(linalg.det(U) < 0):
#        U = U[:,range(1,-1,-1)]
#        V = V[:,range(1,-1,-1)]
#        S = S[:,range(1,-1,-1)]
#        temp[1,1] = S[1]
#        temp[0,0] = S[0]
#        S = temp
#    else:
#        temp[1,1] = S[0]
#        temp[0,0] = S[1]
#        S = temp	
    # New by Jesse
    if(linalg.det(U) < 0):
        U = U[:,range(1,-1,-1)]
        V = V[:,range(1,-1,-1)]
        S = S[:,range(1,-1,-1)]
        S = S[range(1,-1,-1),:]
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
    if phi >= np.pi/2:
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

cdef double[:,:] warpimg(double[:,:] img, double[:,:] p, double[:,:] sz):
    # arrays
    cdef double w,h
    cdef double[:] x
    cdef double[:] y
    cdef double[:,:] result
    #cdef double[:,:] xcoord
    #cdef double[:,:] ycoord
    cdef double[:,:] temp
    cdef double[:] temp1
    w = sz[0,0]
    h = sz[0,1]
    result = np.zeros((w*h,p.shape[0]),dtype=np.float64)
    #May have problem    
    x = np.linspace(1-h/2,h/2,h)
    y = np.linspace(1-w/2,w/2,w)
    xv,yv = np.meshgrid(x,y)
    for i in xrange(p.shape[0]):
        temp = np.concatenate(
            (
                np.ones((w*h,1)),
                xv.reshape((w*h,1),order='F'),
                yv.reshape((w*h,1),order='F')
            ),axis=1)*\
            MA([[p[i,0],p[i,1]],[p[i,2],p[i,4]],[p[i,3],p[i,5]]])
        temp1 = sample_region(img, np.array(temp))
        result[:,i] = temp1[:]
    return result

# TODO
cdef double bilinear_interp(double [:,:] img, double x, double y):
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]

    cdef unsigned int lx = <int>floor(x)
    cdef unsigned int ux = <int>ceil(x)
    cdef unsigned int ly = <int>floor(y)
    cdef unsigned int uy = <int>ceil(y)

    # Need to be a bit careful here due to overflows
    if not (0 <= lx < w and 0 <= ux < w and
            0 <= ly < h and 0 <= uy < h): return 128

    cdef double dx = x - lx
    cdef double dy = y - ly
    return img[ly,lx]*(1-dx)*(1-dy) + \
           img[ly,ux]*dx*(1-dy) + \
           img[uy,lx]*(1-dx)*dy + \
           img[uy,ux]*dx*dy

cdef double[:] sample_region(double[:,:] img, double[:,:] temp):
    cdef int num_pts
    cdef int width
    cdef int height
    cdef int i,j,k
    cdef double w,h
    cdef double[:] result

    num_pts = temp.shape[0]
    height = img.shape[0]    
    width = img.shape[1]
    result = np.empty(num_pts, dtype=np.float64)
    j = 0
    k = 1
    for i in xrange(num_pts):
        w = temp[i,j]
        h = temp[i,k]
        result[i] = bilinear_interp(img, w, h)
    return result
                

def affparam2mat(p):
    # Here I use array instead of matrix
#    cdef double[:] s,th,r,phi
#    cdef double[:] cth,sth,cph,sph
#    cdef double[:] ccc,ccs,css,scc,scs,sss
#    cdef double[:,:]  q
    q = np.zeros(np.asarray(p).shape,'f8')
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

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double[:] est_weight_warp(double[:,:] img, double[:,:] template, double[:,:] param, int n_samples, double[:,:] sz_T, double[:,:] affsig, double alpha):
    cdef int n_process, i, partial
    cdef double[:] wlist
    n_process = 2
    partial = n_samples / n_process
    wlist = np.empty(n_samples, dtype=np.float64)
    return wlist

# additive random walk
def estwarp_condens(img,param, n_sample, sz_T, affsig, MModel):
        update = np.random.randn(n_sample,6) * np.tile(np.array(affsig),[n_sample,1])
        #affsig_used = affsig.copy()
        if MModel == 4:
            1
        elif MModel == 6:
            update[:, 2:] = 0
        elif MModel == 7:
            update[:,4] = update[:,2]
            update[:,5] = 0
        param = param + update
        samples = warpimg(img,affparam2mat(param),sz_T)
        return samples,param

def resample2(curr_samples, prob):
    dt = np.dtype('f8')
    nsamples = MA(curr_samples.shape).item(0)
    afnv = 0
    if prob.sum() == 0:
        map_afnv = MA(np.ones(nsamples), dt).T * afnv
        count = MA(np.zeros((prob.shape), dt))
    else:
        prob = prob / prob.sum()
        N = nsamples
        Ninv = 1 / float(N)
        map_afnv = MA(np.zeros((N, 6)), dt)
        c = pylab.cumsum(prob)
        u = pylab.rand() * Ninv
        i = 0
        # pdb.set_trace()
        for j1 in range(N):
            uj = u + Ninv * j1
            while uj > c[i]:
                i += 1
            map_afnv[j1, :] = curr_samples[i, :]
        return map_afnv

cdef double [:] weight_eval(double[:,:] samples, double[:,:] template, double[:,:] sz, double alpha, int use_scv, double[:] intensity_map, double[:,:] template_old):
    #wlist = [i for i in range(samples.shape[1])]
    cdef int i
    cdef double wlist_u
    cdef double [:] wlist
    cdef double [:] temp_norm
    cdef double [:,:] temp_sample
    cdef double [:,:] temp_template
    wlist = np.zeros(samples.shape[1])
    samples_copy = samples.copy()
    if use_scv != 0:
        for indx in xrange(samples_copy.shape[1]):
            if intensity_map == None:
                intensity_map = scv_intensity_map(samples_copy[:,indx], template_old[:,0])
            samples_copy[:, indx] = scv_expected_img(samples_copy[:,indx], intensity_map)
    temp_samples  = whiten(samples_copy)
    temp_samples = normalizeTemplates(temp_samples)
    temp_template = np.asarray(template).reshape((sz[0,0]*sz[0,1],1),order='F')
    for i in xrange(temp_samples.shape[1]):
        temp_norm = np.array(temp_samples[:,i]) - np.array(temp_template[:,0])
        wlist_u = np.linalg.norm(temp_norm)
        wlist[i] = np.exp(-alpha*(wlist_u*wlist_u))
        if str(wlist[i]) == 'nan':pdb.set_trace()    
    return wlist

def drawbox(est,sz):
    temp = np.zeros(6)
    temp[:] = est[0,:]
    est = temp
    M = np.array([[est[0],est[2],est[3]],[est[1],est[4],est[5]]])
    w = sz[0,0]
    h = sz[0,1]
    corners = np.array([[1,-w/2,-h/2],[1,w/2,-h/2],[1,w/2,h/2],[1,-w/2,h/2]]).T
    corners = MA(M) * MA(corners)
    return corners

# center a set of images by subtracting the mean and dividing the std
cdef double[:,:] whiten(double[:,:] In):
    cdef int MN
    cdef double[:,:] out
    dt = np.dtype('f8')
    MN = np.asarray(In).shape[0]
    a = np.mean(In,axis=0)
    b = np.std(In,axis=0) + 1e-14
    out = np.divide( ( In - (MA(np.ones((MN,1)),dt)*a)), (MA(np.ones((MN,1)),dt) *b) )
    return out

def des_sort(q):
        temp = q
        B1 = numpy.sort(q);
        Asort = B1[::-1];
        B = numpy.argsort(q)
        A_ind= B[::-1]
        return Asort, A_ind

cdef double[:,:] normalizeTemplates(double[:,:] A):
        cdef int MN
        cdef double[:] A_norm
        MN = A.shape[0]
        A_norm = np.sqrt(np.sum(np.multiply(A,A),axis=0)) + 1e-14;
        A = np.divide(A,(np.ones((MN,1))*A_norm));
        return A
