__author__ = 'Tommy'
import cv2
import numpy as np


class FilteringParams:
    def __init__(self, type, params):
        self.type = type
        self.validated = False
        self.init_success = True
        self.update = lambda: None
        self.validate = lambda: True

        # initialize parameters
        self.params = {}
        for key in params.keys():
            # print 'key=', key
            default_vals = params[key]['default']
            self.params[key] = Param(name=key, id=params[key]['id'], base=default_vals['base'],
                                     mult=default_vals['mult'],
                                     limit=default_vals['limit'], add=default_vals['add'],
                                     type=params[key]['type'])
        self.sorted_params = self.getSortedParams()
        if type == 'none':
            self.apply = lambda img: img
        elif type == 'gabor':
            # print 'Initializing Gabor filter'
            self.update = lambda: cv2.getGaborKernel(ksize=(self.params['ksize'].val, self.params['ksize'].val),
                                                     sigma=self.params['sigma'].val, theta=self.params['theta'].val,
                                                     lambd=self.params['lambd'].val, gamma=self.params['gamma'].val)
            self.apply = lambda img: cv2.filter2D(img.astype(np.float64), -1, self.kernel)
        elif type == 'laplacian':
            # print 'Initializing Laplacian filter'
            self.apply = lambda img: cv2.Laplacian(img.astype(np.float64), -1, ksize=self.params['ksize'].val,
                                                   scale=self.params['scale'].val, delta=self.params['delta'].val)
        elif type == 'sobel':
            # print 'Initializing Sobel filter'
            self.apply = lambda img: cv2.Sobel(img.astype(np.float64), -1, ksize=self.params['ksize'].val,
                                               scale=self.params['scale'].val,
                                               delta=self.params['delta'].val, dx=self.params['dx'].val,
                                               dy=self.params['dy'].val)
            self.validate = lambda: validate()

            def validate():
                # print 'Validating Sobel derivatives...'
                # print 'dx=',self.params[-2].val, ' dy=', self.params[-1].val
                if self.params['dy'].val == 0 and self.params['dx'].val == 0:
                    #print self.params[-1].name, "is ", self.params[-1].val, " while ", self.params[-2].name, " is ", self.params[-2].val
                    self.validated = False
                    return False
                if self.params['dy'].val >= self.params['ksize'].val:
                    #print self.params[0].name, "is ", self.params[0].val, " while ", self.params[-1].name, " is ", self.params[-1].val
                    self.validated = False
                    return False
                if self.params['dx'].val >= self.params['ksize'].val:
                    #print self.params[0].name, "is ", self.params[0].val, " while ", self.params[-2].name, " is ", self.params[-2].val
                    self.validated = False
                    return False
                return True
        elif type == 'scharr':
            # print 'Initializing Scharr filter'
            self.update = lambda: validate()
            self.apply = lambda img: cv2.Scharr(img.astype(np.float64), -1, scale=self.params['scale'].val,
                                                delta=self.params['delta'].val, dx=self.params['dx'].val,
                                                dy=self.params['dy'].val)
            self.validate = lambda: validate()

            def validate():
                # print 'Validating Scharr derivatives...'
                # print 'dx=',self.params[-2].val, ' dy=', self.params[-1].val
                if self.params['dy'].val == 0 and self.params['dx'].val == 0:
                    self.validated = True
                    return False
                if self.params['dy'].val + self.params['dx'].val > 1:
                    self.validated = True
                    return False
                return True
        elif type == 'canny':
            # print 'Initializing Canny Edge Detector'
            self.apply = lambda img: applyCanny(img)

            def applyCanny(img):
                if len(img.shape) == 3:
                    canny_img = np.empty(img.shape)
                    for i in xrange(img.shape[2]):
                        canny_img[:, :, i] = cv2.Canny(img[:, :, i], threshold1=self.params['low_thresh'].val,
                                                       threshold2=self.params['low_thresh'].val * self.params[
                                                           'ratio'].val)
                    return canny_img
                else:
                    return cv2.Canny(img, threshold1=self.params['low_thresh'].val,
                                     threshold2=self.params['low_thresh'].val * self.params['ratio'].val)
        elif type == 'DoG':
            # print 'Initializing DoG filter'
            self.apply = lambda img: applyDoG(img)

            def applyDoG(img):
                img = img.astype(np.uint8)
                ex_img = cv2.GaussianBlur(img, ksize=(self.params['ksize'].val, self.params['ksize'].val),
                                          sigmaX=self.params['exc_std'].val)
                in_img = cv2.GaussianBlur(img, (self.params['ksize'].val, self.params['ksize'].val),
                                          sigmaX=self.params['inh_std'].val)
                dog_img = ex_img - self.params['ratio'].val * in_img
                dog_img = dog_img.astype(np.uint8)
                return dog_img
        elif type == 'LoG':
            # print 'Initializing LoG filter'
            self.apply = lambda img: applyLoG(img)

            def applyLoG(img):
                img = img.astype(np.float64)
                gauss_img = cv2.GaussianBlur(img,
                                             ksize=(self.params['gauss_ksize'].val, self.params['gauss_ksize'].val),
                                             sigmaX=self.params['std'].val)
                log_img = cv2.Laplacian(gauss_img, -1, ksize=self.params['lap_ksize'].val,
                                        scale=self.params['scale'].val, delta=self.params['delta'].val)
                log_img = log_img.astype(np.uint8)
                return log_img
        else:
            self.init_success = False
            print "Invalid filter type:", type

        # self.printParamValue()
        self.kernel = self.update()
        # print 'in FilterParams __init__ kernel=', self.kernel

    def getSortedParams(self):
        return sorted(self.params.values(), key=lambda k: k.id)

    def printParamValue(self):
        print 'Initialized', self.type.capitalize(), 'filter with:'
        for key in self.params.keys():
            print key, "=", self.params[key].val
        print "\n" + "*" * 60 + "\n"


class Param:
    def __init__(self, name, id, base, mult, limit, add=0.0, type='float'):
        self.name = name
        self.id = id
        self.base = base
        self.multiplier = mult
        self.add = add
        self.limit = limit
        self.val = 0
        self.type = type
        self.updateValue()

    def updateValue(self, multiplier=None):
        if multiplier != None:
            self.multiplier = multiplier
        self.val = self.base * self.multiplier + self.add
        if self.type == 'int':
            self.val = int(self.val)
        if multiplier != None:
            print self.name, "updated to", self.val

