__author__ = 'abhineet'
import CModules.xvSSDTransPy as xvSSDTrans
import CModules.xvSSDRotatePy as xvSSDRotate
import CModules.xvSSDRTPy as xvSSDRT
import CModules.xvSSDSE2Py as xvSSDSE2
import CModules.xvSSDPyramidTransPy as xvSSDPyramidTrans
import CModules.xvSSDPyramidRotatePy as xvSSDPyramidRotate
import CModules.xvSSDPyramidRTPy as xvSSDPyramidRT
import CModules.xvSSDPyramidSE2Py as xvSSDPyramidSE2

from TrackerBase import *

class XVSSDTracker(TrackerBase):
    def __init__(self, steps_per_frame, multi_approach, enable_scv, direct_capture, stepper,
                 use_pyramidal_stepper, show_xv_window, no_of_levels=2, scale=0.5):
        print "Initializing XVSSD Tracker  with:"
        print "\t steps_per_frame=", steps_per_frame
        print "\t multi_approach=", multi_approach
        print "\t stepper=", stepper
        print "\t use_pyramidal_stepper=", use_pyramidal_stepper
        print "\t show_xv_window=", show_xv_window
        print "\t direct_capture=", direct_capture
        print "\t no_of_levels=", no_of_levels
        print "\t scale=", scale


        self.steps_per_frame = steps_per_frame
        self.size_x = 0
        self.size_y = 0
        self.proposal = None
        self.multi_approach = multi_approach
        self.initialized = False
        self.enable_scv = enable_scv
        self.direct_capture = direct_capture
        self.show_xv_window = show_xv_window
        self.stepper = stepper
        self.use_pyramidal_stepper = use_pyramidal_stepper
        self.no_of_levels = int(no_of_levels)
        self.scale = scale
        self.text='xv_ssd'
        if self.use_pyramidal_stepper:
            self.text = self.text + '_' + 'pyramidal'
        self.text = self.text + '_' + self.stepper

        if self.stepper == 'trans':
            if self.use_pyramidal_stepper:
                self.tracker = xvSSDPyramidTrans
            else:
                self.tracker = xvSSDTrans
        elif self.stepper == 'se2':
            if self.use_pyramidal_stepper:
                self.tracker = xvSSDPyramidSE2
            else:
                self.tracker = xvSSDSE2
        elif self.stepper == 'rotate':
            if self.use_pyramidal_stepper:
                self.tracker = xvSSDPyramidRotate
            else:
                self.tracker = xvSSDRotate
        elif self.stepper == 'rt':
            if self.use_pyramidal_stepper:
                self.tracker = xvSSDPyramidRT
            else:
                self.tracker = xvSSDRT
        else:
            raise SystemExit('Invalid stepper specified')


    def initialize(self, img, region):
        if len(img.shape) != 3:
            raise SystemExit('Error in XVSSDTracker: '
                             'Expected multi channel image but found a single channel one')
        self.proposal = np.copy(region)

        center_x = (region[0, 0] + region[0, 2]) / 2
        center_y = (region[1, 0] + region[1, 2]) / 2

        # obj_pos = np.array([center_x, center_y], dtype=np.float64)
        # print 'obj_pos:\n', obj_pos

        self.size_x = abs(region[0, 0] - region[0, 2])
        self.size_y = abs(region[1, 0] - region[1, 2])

        # obj_size = np.array([self.size_x, self.size_y], dtype=np.float64)
        # print 'obj_size:\n', obj_size

        if self.use_pyramidal_stepper:
            self.tracker.initialize(img.astype(np.uint8), center_x, center_y, self.size_x, self.size_y,
                                    self.steps_per_frame, self.no_of_levels, self.scale,
                                    self.direct_capture, self.show_xv_window)
        else:
            self.tracker.initialize(img.astype(np.uint8), center_x, center_y, self.size_x, self.size_y,
                                    self.steps_per_frame, self.direct_capture, self.show_xv_window)

        # if self.stepper == 'trans':
        # xvSSDTrans.initialize(img.astype(np.uint8), center_x, center_y, self.size_x, self.size_y,
        # self.steps_per_frame, self.direct_capture)
        # elif self.stepper == 'se2':
        # xvSSDSE2.initialize(img.astype(np.uint8), center_x, center_y, self.size_x, self.size_y,
        #                         self.steps_per_frame, self.direct_capture)
        # else:
        #     raise SystemExit('Invalid stepper specified')

        self.initialized = True

    def update(self, img):
        # print 'in XVSSDTransTracker: update'

        self.proposal = self.tracker.update(img.astype(np.uint8))

        # if self.stepper == 'trans':
        # self.proposal = xvSSDTrans.update(img.astype(np.uint8))
        # elif self.stepper == 'se2':
        # self.proposal = xvSSDSE2.update(img.astype(np.uint8))
        # else:
        # raise SystemExit('Invalid stepper specified')


        # print 'XVSSDTransTracker: corners:\n', self.proposal
        # min_y = pos_y - self.size_y / 2
        #
        # max_x = pos_x + self.size_x / 2
        # max_y = pos_y + self.size_y / 2

        # self.proposal = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]], dtype=np.float64).T


    def get_region(self):
        return self.proposal


    def set_region(self, corners):
        pass


    def cleanup(self):
        pass


    def is_initialized(self):
        return self.initialized








