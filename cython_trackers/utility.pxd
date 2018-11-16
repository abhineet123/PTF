cdef double bilin_interp(double [:,:] img, double x, double y)

cpdef double[:] sample_pts_all(double[:,:] img, int resx, int resy, double[:,:] warp, int MModel, double[:,:] tmplt_size = ?) except *
cpdef double[:] sample_pts(double[:,:] img, int resx, int resy, double[:,:] warp) except *

cpdef double[:,:] make_hom_sl3(double[:] p) except *
cpdef double[:,:] aff_update_backward(double[:,:] warp, double[:] update) except *

cpdef double[:,:] sample_pts_grad_sl3(double[:,:] img, int resx, int resy, double[:,:] warp) except *

cpdef double[:,:] sample_pts_Jacob(double[:,:] img, int resx, int resy, double[:,:] warp, int MModel, double[:,:] tmplt_size = ?) except *

cpdef double [:,:] to_grayscale(unsigned char [:,:,:] img)

cdef normalize_hom(double[:,:] m)
cdef double[:,:] mat_mul(double[:,:] A, double[:,:] B) 
#cdef double[:] mat_min(double[:] A, double[:] B)
cdef double[:,:] warp_update(double[:,:] old, double [:] update)

cdef double[:] scv_intensity_map(double[:] img1, double[:] img2)
cdef double[:] scv_expected_img(double[:] img, double[:] intensity_map)

cpdef double[:,:] compute_homography(double[:,:] in_pts, double[:,:] out_pts)
cpdef double[:,:] compute_affine(double[:,:] in_pts, double[:,:] tmplt_size, double[:,:] out_pts)

# For storing the pixels!!! 

