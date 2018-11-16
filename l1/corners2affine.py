import numpy
from numpy import matrix as MA


def corners2affine(corners_in, size_out):
    dt = numpy.dtype('f8')
    rows = size_out.item(0)
    cols = size_out.item(1)
    inp = numpy.insert(corners_in, 2, numpy.matrix([[1, 1, 1]], dt), 0)
    outp = MA([[1, rows, 1], [1, 1, cols], [1, 1, 1]], dt)
    R = inp * outp.I
    afnv = MA([[R.item(0, 0), R.item(0, 1), R.item(1, 0), R.item(1, 1), R.item(0, 2), R.item(1, 2)]], dt)
    afnv_obj = {'R': R, 'afnv': afnv, 'size_out': size_out}
    return afnv_obj
