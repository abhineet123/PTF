import numpy
import scipy.ndimage
from numpy import matrix as MA
import math

from IMGaffine import IMGaffine_c
from softresh import softresh_c
from APGLASSOup import APGLASSOup_c

def des_sort(q):
    B1 = numpy.sort(q);
    Asort = B1[::-1];
    B = numpy.argsort(q)
    A_ind = B[::-1]
    return Asort, A_ind


def replacement(X, n):
    X = X.flatten(1)
    count = 0;
    for i in range(X.shape[0]):
        if X[i] == n:
            count = count + 1;
            X[i] = 0
    return count


def labels(x):
    (X, T) = scipy.ndimage.label(x)
    areastat = MA(numpy.zeros((T, 2)))
    for j in range(1, T + 1):
        count = replacement(X, j)
        areastat[j - 1, :] = [j, count];

    return areastat, T


def binary(img1, level):
    img1 = MA(img1)
    tempIm = img1.flatten(1)
    for i in range(tempIm.shape[1]):
        if tempIm[0, i] < level:
            tempIm[0, i] = 0;
        else:
            tempIm[0, i] = 1;
    tempIm = (numpy.reshape(tempIm, (img1.shape[1], img1.shape[0]))).T
    return tempIm


def draw_sample(mean_afnv, std_afnv):
    dt = numpy.dtype('f8')
    mean_afnv = MA((mean_afnv), dt);
    nsamples = MA(mean_afnv.shape).item(0)
    MV_LEN = 6
    mean_afnv[:, 0] = numpy.log(mean_afnv[:, 0])
    mean_afnv[:, 3] = numpy.log(mean_afnv[:, 3])
    outs = MA(numpy.zeros((nsamples, MV_LEN)), dt)
    flatdiagonal = MA((numpy.diagflat(std_afnv)), dt);
    outs[:, 0:MV_LEN] = MA(numpy.random.randn(nsamples, MV_LEN), dt) * flatdiagonal + mean_afnv
    outs[:, 0] = MA(numpy.exp(outs[:, 0]), dt)
    outs[:, 3] = MA(numpy.exp(outs[:, 3]), dt)
    return outs


def crop_candidates(img_frame, curr_samples, template_size):
    dt = numpy.dtype('f8')
    nsamples = MA(curr_samples.shape).item(0)
    c = numpy.prod(template_size)
    gly_inrange = MA(numpy.zeros(nsamples), dt)
    gly_crop = MA(numpy.zeros((c, nsamples)), dt)
    for i in range(nsamples):
        curr_afnv = curr_samples[i, :]
        img_cut, gly_inrange[0, i] = IMGaffine_c(img_frame, curr_afnv, template_size)
        gly_crop[:, i] = MA(img_cut).flatten(1).T
    return gly_crop, gly_inrange


def normalizeTemplates(A):
    MN = A.shape[0]
    A_norm = numpy.sqrt(numpy.sum(numpy.multiply(A, A), axis=0)) + 1e-14;
    A = numpy.divide(A, (numpy.ones((MN, 1)) * A_norm));
    return A, A_norm


def resample(curr_samples, prob, afnv):
    dt = numpy.dtype('f8')
    nsamples = MA(curr_samples.shape).item(0)
    if prob.sum() == 0:
        map_afnv = MA(numpy.ones(nsamples), dt) * afnv
        count = MA(numpy.zeros((prob.shape), dt))
    else:
        prob = prob / (prob.sum())
        count = MA(numpy.ceil(nsamples * prob), int)
        count = count.T
        map_afnv = MA(numpy.zeros((1, 6)), dt);
        for i in range(nsamples):
            for j in range(count[i]):
                map_afnv = MA(numpy.concatenate((map_afnv, curr_samples[i, :]), axis=0), dt)
        K = map_afnv.shape[0];
        map_afnv = map_afnv[1:K, :];
        ns = count.sum()
        if nsamples > ns:
            map_afnv = MA(numpy.concatenate((map_afnv, MA(numpy.ones(((nsamples - ns), 1)) * afnv, dt)), axis=0), dt)
        map_afnv = map_afnv[0:nsamples, :]
    return map_afnv, count


def images_angle(I1, I2):
    I1v = I1.flatten(1)
    col = MA(I1v.shape).item(1);
    I2v = I2.flatten(1)
    I1vn = I1v / (numpy.sqrt(numpy.sum(numpy.multiply(I1v, I1v))) + 1e-14);
    I2vn = I2v / (numpy.sqrt(numpy.sum(numpy.multiply(I2v, I2v))) + 1e-14);
    angle = math.degrees(math.acos(numpy.sum(numpy.multiply(I1vn, I2vn))))
    return angle


def whiten(In):
    dt = numpy.dtype('f8')
    MN = MA(In.shape).item(0)
    a = numpy.mean(In, axis=0)
    b = numpy.std(In, axis=0) + 1e-14
    out = numpy.divide(( In - (MA(numpy.ones((MN, 1)), dt) * a)), (MA(numpy.ones((MN, 1)), dt) * b))
    return out, a, b

