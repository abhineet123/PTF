__author__ = 'Tommy'

from TrackerBase import *
import time
import numpy as np
from l1.InitTemplates import InitTemplates
from l1.corners2affine import corners2affine
from l1.affine2image import aff2image
from l1.misc import *

class L1Tracker(TrackerBase):
    def __init__(self, no_of_samples, angle_threshold, res, no_of_templates, alpha,
                 multi_approach='none', use_scv=False, feature='none'):

        print "Initializing L1 tracker with:"
        print " no_of_samples=", no_of_samples
        print " angle_threshold=", angle_threshold
        print " res=", res
        print " no_of_templates=", no_of_templates
        print " alpha=", alpha
        print " multi_approach=", multi_approach
        print " feature=", feature

        self.n_sample = no_of_samples
        self.angle_threshold = angle_threshold
        self.no_of_templates = no_of_templates
        self.alpha = alpha
        self.feature=feature

        self.Lambda = np.matrix([[0.2, 0.001, 10]])
        resx = res[0]
        resy = res[1]
        if resy<=0:
            resy=resx
        self.res=(resx, resy)
        self.sz_T = np.matrix(self.res)
        self.rel_std_afnv = np.matrix([[0.005, 0.003, 0.005, 0.003, 1, 1]])
        self.Lip = 8
        self.Maxit = 5
        self.dt = np.dtype('f8')
        #self.dt2 = np.dtype('uint8')
        #self.dt3 = np.dtype('float64')
        self.initialized = False
        self.multi_approach=multi_approach
        self.use_scv=use_scv

        self.occlusionNf = 0

        # misc variables set to None
        self.cmax = None
        self.A = None
        self.T = None
        self.map_aff = None
        self.aff_samples = None
        self.fixT = None
        self.Temp = None
        self.Dict = None
        self.Temp1 = None
        self.position=None
        self.corners=None

    def set_region(self, region):

        self.corners=np.matrix(
            [[int(region[0, 0]), int(region[1, 0]), int(region[2, 0]), int(region[3, 0])],
             [int(region[0, 1]), int(region[1, 1]), int(region[2, 1]), int(region[3, 1])]]
        )
        #corners=np.array([[int(inp[1, 0]), int(inp[0, 0])],
        # [int(inp[1, 2]), int(inp[0, 2])],
        # [int(inp[1, 3]), int(inp[0, 3])],
        # [int(inp[1, 1]), int(inp[0, 1])]])
        self.position=np.matrix(
            [[int(region[0, 1]), int(region[3, 1]), int(region[1, 1])],
             [int(region[0, 0]), int(region[3, 0]), int(region[1, 0])]]
        )
        #print "in set_region"
        #print "region=", region
        #print "self.corners=", self.corners

    def initialize(self, img, region):
        self.set_region(region.T)
        #init_pos = np.matrix(region)
        img = np.matrix(img)
        (self.T, T_norm, T_mean, T_std) = InitTemplates(self.sz_T, self.no_of_templates, img, self.position)
        #self.norms = np.multiply(T_norm, T_std) # %template norms

        # L1 function settings
        dim_T = self.sz_T[0, 0] * self.sz_T[0, 1]    # number of elements in one template, sz_T(1)*sz_T(2)=12x15 = 180
         # data matrix is composed of T, positive trivial T.
        self.A = np.matrix(np.concatenate((self.T, np.matrix(np.identity(dim_T))), axis=1))
         # get affine transformation parameters from the corner points in the first frame
        (aff_obj) = corners2affine(self.position, self.sz_T)
        self.map_aff = aff_obj['afnv']
        self.aff_samples = np.dot(np.ones((self.n_sample, 1), self.dt), self.map_aff)

        #T_id = -np.arange(self.no_of_templates)   # % template IDs, for debugging

        self.fixT = self.T[:, 0] / self.no_of_templates #  first template is used as a fixed template  CHECK THIS

        # Temaplate Matrix
        self.Temp = np.concatenate((self.A, self.fixT), axis=1)
        self.Dict = np.dot(self.Temp.T, self.Temp)

        temp1 = np.concatenate((self.T, self.fixT), axis=1)
        self.Temp1 = temp1 * np.linalg.pinv(temp1)

        #temp1 = np.concatenate((self.A, self.fixT), axis=1)
        #colDim = np.matrix(temp1.shape).item(1)
        #self.Coeff = np.zeros((colDim, self.nframes), self.dt)

        self.initialized = True

    def update(self, img):
        start_time = time.time()
        # Draw transformation samples from a Gaussian distribution
        temp_map_aff = np.sum(np.multiply(self.map_aff[0, 0:4], self.map_aff[0, 0:4])) / 2
        sc = np.sqrt(temp_map_aff)
        std_aff = np.multiply(self.rel_std_afnv, np.matrix([[1, sc, sc, 1, sc, sc]]))

        self.map_aff = self.map_aff + 1e-14
        (self.aff_samples) = draw_sample(self.aff_samples,
                                         std_aff) # draw transformation samples from a Gaussian distribution

        (Y, Y_inrange) = crop_candidates(img, self.aff_samples[:, 0:6], self.sz_T)

        if np.sum(Y_inrange == 0) == self.n_sample:
            print 'Target is out of the frame!\n'

        (Y, Y_crop_mean, Y_crop_std) = whiten(Y)     # zero-mean-unit-variance
        (Y, Y_crop_norm) = normalizeTemplates(Y) # norm one
        #%-L1-LS for each candidate target

        q = np.zeros((self.n_sample, 1), self.dt) #  % minimal error bound initialization
        # % first stage L2-norm bounding
        for j in range(self.n_sample):
            cond1 = Y_inrange[0, j]
            temp_abs = np.absolute(Y[:, j])
            cond2 = np.sum(temp_abs)
            if cond1 == 0 and cond2 == 0:
                continue
                # L2 norm bounding
            temp_x_norm = Y[:, j] - self.Temp1 * Y[:, j]
            q[j, 0] = np.linalg.norm(temp_x_norm)
            q[j, 0] = np.exp(-self.alpha * (q[j, 0] * q[j, 0]))

        #-------------------------------------------------------------------------------------------------
        eta_max = float("-inf")
        # sort samples according to descend order of q
        (qtemp1, indqtemp) = des_sort(q.T)
        q = qtemp1.T
        indq = indqtemp.T

        #second stage
        p = np.zeros((self.n_sample), self.dt) #observation likelihood initialization
        n = 0
        id_max = indq[n]
        tau = 0
        while (n < self.n_sample) and (q[n] >= tau):
            APG_arg1 = (self.Temp.T * Y[:, indq[n]])
            (c) = APGLASSOup_c(APG_arg1, self.Dict, self.Lambda, self.Lip, self.Maxit, self.no_of_templates)
            c = np.matrix(c)
            Ele1 = np.concatenate((self.A[:, 0:self.no_of_templates], self.fixT), axis=1)
            Ele2 = np.concatenate((c[0:self.no_of_templates], c[-1]), axis=0)
            D_s = (Y[:, indq[n - 1]] - Ele1 * Ele2) #reconstruction error
            D_s = np.multiply(D_s, D_s) #reconstruction error
            p[indq[n]] = np.exp(-self.alpha * (np.sum(D_s))) #  probability w.r.t samples
            tau += p[indq[n]] / (2 * self.n_sample - 1)# update the threshold
            if p[indq[n]] > eta_max: #******POssilbe Erro*****
                id_max = indq[n]
                self.c_max = c
                eta_max = p[indq[n]]
            n += 1
        self.map_aff = self.aff_samples[id_max, 0:6] # target transformation parameters with the maximum probability
        a_max = self.c_max[0:self.no_of_templates, 0]
        (self.aff_samples, _) = resample(self.aff_samples, p, self.map_aff) # resample the samples wrt. the probability
        indA = a_max.argmax()
        min_angle = images_angle(Y[:, id_max], self.A[:, indA])
        # -Template update
        self.occlusionNf = self.occlusionNf - 1
        level = 0.05
        Initialparameterlambda = np.matrix(self.Lambda)

        if min_angle > self.angle_threshold and self.occlusionNf < 0:
            print ('Update!')
            trivial_coef = (np.reshape(self.c_max[self.no_of_templates:-1, 0], (self.sz_T[0, 1], self.sz_T[0, 0]))).T
            #dst1 = np.matrix(np.zeros((self.sz_T[0, 0], self.sz_T[0, 1])))
            trivial_coef = binary(trivial_coef, level)
            #se = np.matrix([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0],
            #                [0, 1, 1, 1, 0], [0, 0, 1, 0, 0],
            #                [0, 0, 0, 0, 0]])
            #se = np.array(se.T)
            areastats, T1 = labels(trivial_coef)
            if T1 > 0:
                Area = areastats[:, 1]
                max_area = Area.max()

            Area_tolerance = 0.25 * self.sz_T[0, 0] * self.sz_T[0, 1]
            # Occlusion Detection
            if T1 > 0 and max_area < np.rint(Area_tolerance):
            # find the template to be replaceed

                tempa_max = a_max[0:self.no_of_templates - 1, 0]
                indW = tempa_max.argmin()

                # insert new template
                self.T = np.matrix(self.T)

                #self.T_id = np.matrix(self.T_id)
                #self.T_mean = np.matrix(self.T_mean)
                #self.norms = np.matrix(self.norms)
                #
                self.T[:, indW] = Y[:, id_max]

                #self.T_mean[indW, 0] = Y_crop_mean[0, indW]
                #self.T_id[0, indW] = t - 1 # track the replaced template for debugging
                #self.norms[indW, 0] = Y_crop_std[0, id_max] * Y_crop_norm[0, id_max];

                (self.T, _) = normalizeTemplates(self.T)
                self.A[:, 0:self.no_of_templates] = self.T

                # Template Matrix
                self.Temp = np.matrix(np.concatenate((self.A, self.fixT), axis=1))
                self.Dict = self.Temp.T * self.Temp
                tempInverse = np.concatenate((self.T, self.fixT), axis=1)
                self.Temp1 = tempInverse * np.linalg.pinv(tempInverse)
            else:
                #occlusion = 5
                # update L2 regularized  term
                self.Lambda = np.matrix([[Initialparameterlambda[0, 0], Initialparameterlambda[0, 1], 0]]);

        elif self.occlusionNf < 0:
            self.Lambda = Initialparameterlambda

        rect = np.rint(aff2image(self.map_aff.T, self.sz_T))
        inp = (np.reshape(rect, (4, 2))).T
        #point1 = (int(inp[1, 0]), int(inp[0, 0]))
        #point2 = (int(inp[1, 2]), int(inp[0, 2]))
        #point3 = (int(inp[1, 3]), int(inp[0, 3]))
        #point4 = (int(inp[1, 1]), int(inp[0, 1]))
        corners=np.array([[int(inp[1, 0]), int(inp[0, 0])],
                 [int(inp[1, 2]), int(inp[0, 2])],
                 [int(inp[1, 3]), int(inp[0, 3])],
                 [int(inp[1, 1]), int(inp[0, 1])]])

        self.set_region(corners)
        #self.position = np.matrix(
        #    [[int(inp[1, 0]), int(inp[0, 0]), int(inp[1, 3] - inp[1, 0]), int(inp[0, 3] - inp[0, 0])]])
        #self.position=self.position.T

    def is_initialized(self):
        return self.initialized

    def get_region(self):
        return self.corners

    def cleanup(self):
        pass





