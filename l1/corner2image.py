import numpy
from whitening import whiten
from IMGaffine import IMGaffine_c
from numpy import matrix as MA
from numpy import linalg
from corners2affine import corners2affine


def corner2image(img, p, tsize):
    dt3 = numpy.dtype('float64')
    afnv_obj = corners2affine(p, tsize)
    map_afnv = afnv_obj['afnv']
    img = MA((img), dt3);
    img_map, blah = IMGaffine_c(img, map_afnv, tsize)
    crop, crop_mean, crop_std = whiten(img_map.flatten(1).T) # reshape different from matlab
    crop_norm = linalg.norm(crop)
    crop = crop / crop_norm
    return crop, crop_norm, crop_mean, crop_std
