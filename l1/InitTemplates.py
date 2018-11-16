import numpy
from numpy import matrix as MA
from corner2image import corner2image


def InitTemplates(tsize, numT, img, cpt):
    dt = numpy.dtype('f8')
    p = []
    p.append(cpt)
    for i in range(numT - 1):
        p.append(cpt + MA(numpy.random.randn(2, 3)) * 0.6)
    A = numpy.prod(tsize)
    T = numpy.zeros((A, numT))
    T = MA(T, dt)
    T_norm = MA(numpy.zeros((numT, 1)), dt)
    T_mean = MA(numpy.zeros((numT, 1)), dt)
    T_std = MA(numpy.zeros((numT, 1)), dt)
    for i in range(numT):
        A1, T_norm[i], T_mean[i], T_std[i] = corner2image(img, p[i], tsize)
        T[:, i] = A1[:, 0]
    return T, T_norm, T_mean, T_std
