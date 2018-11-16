__author__ = 'Tommy'
import cv2
import numpy as np
from MultiProposalTracker import MultiProposalTracker
from ParallelTracker import ParallelTracker
from BakerMatthewsICTracker import BakerMatthewsICTracker as BMICTracker
from CascadeTracker import CascadeTracker
from ESMTracker import ESMTracker
from L1Tracker import L1Tracker

flann_found = True
cython_found = True
xvision_found = True
mtf_found = True
try:
    from NNTracker import NNTracker
except ImportError:
    print 'FLANN not found'
    flann_found = False
try:
    from cython_trackers.BMICTracker import BMICTracker as BMICTracker_c
    from cython_trackers.CascadeTracker import CascadeTracker as CascadeTracker_c
    from cython_trackers.ESMTracker import ESMTracker as ESMTracker_c
    from cython_trackers.CTracker import CTracker as RKLTracker
    from cython_trackers.PFTracker import PFTracker as PFTracker
    from cython_trackers.DESMTracker import DESMTracker as DESMTracker
    from cython_trackers.DLKTracker import DLKTracker as DLKTracker
except ImportError:
    print 'Cython modules not found'
    cython_found = False
if cython_found and flann_found:
    from cython_trackers.NNTracker import NNTracker as NNTracker_c
    from cython_trackers.DNNTracker import DNNTracker as DNNTracker
    import cython_trackers.TurnkeyTrackers as cy_tt
try:
    from XVSSDTracker import XVSSDTracker
except ImportError:
    print 'Xvision not found'
    xvision_found = False
try:
    from mtfTracker import mtfTracker
except ImportError as err:
    print 'MTF not found: {:s}'.format(err)
    mtf_found = False
from Homography import random_homography

class TrackingParams:
    def __init__(self, type, params):
        self.type = type
        self.params = {}
        self.tracker = None

        self.update = lambda: None
        self.validate = lambda: True
        # print 'params.keys=\n', params.keys()

        self.mmodels = {
            8: 1,
            6: 4,
            4: 7,
            3: 5,
            2: 6}

        # initialize parameters
        for key in params.keys():
            vals = params[key]
            self.params[key] = Param(name=key, id=vals['id'], val=vals['default'], type=vals['type'],
                                     list=vals['list'])
        self.sorted_params = self.getSortedParams()
        for param in self.sorted_params:
            self.params[param.name] = param


        if flann_found and type == 'nn':
            def getNNTracker(feature, current_params=self.params):
                version = current_params['version'].val
                if version == 'python':
                    print 'Using python version'
                    return NNTracker(
                        no_of_samples=current_params['no_of_samples'].val,
                        no_of_iterations=current_params['no_of_iterations'].val,
                        res=(current_params['resolution_x'].val,
                             current_params['resolution_y'].val),
                        use_scv=current_params['enable_scv'].val,
                        multi_approach=current_params['multi_approach'].val,
                        warp_generator=lambda: random_homography(current_params['nn_sigma_d'].val,
                                                                 current_params['nn_sigma_t'].val),
                        feature=feature)
                elif cython_found and version == 'cython':
                    print 'Using cython version'
                    # NNTracker(nn_iters, nn_samples, res[0], res[1], 0.06, 0.04, use_scv)
                    return NNTracker_c(
                        current_params['no_of_iterations'].val,
                        current_params['no_of_samples'].val,
                        current_params['resolution_x'].val,
                        current_params['resolution_y'].val,
                        current_params['sigma_t'].val,
                        current_params['sigma_d'].val,
                        current_params['enable_scv'].val
                    )

            self.update = getNNTracker
        elif type == 'esm':
            def getESMTracker(feature, current_params=self.params):
                version = current_params['version'].val
                if version == 'python':
                    print 'Using python version'
                    return ESMTracker(
                        max_iterations=current_params['max_iterations'].val,
                        threshold=current_params['threshold'].val,
                        err_thresh=current_params['err_thresh'].val,
                        res=(current_params['resolution_x'].val,
                             current_params['resolution_y'].val),
                        use_err_thresh=current_params['enable_err_thresh'].val,
                        use_scv=current_params['enable_scv'].val,
                        multi_approach=current_params['multi_approach'].val,
                        feature=feature)
                elif cython_found and version == 'cython':
                    print 'Using cython version'
                    # NNTracker(nn_iters, nn_samples, res[0], res[1], 0.06, 0.04, use_scv)
                    return ESMTracker_c(
                        current_params['max_iterations'].val,
                        current_params['threshold'].val,
                        current_params['resolution_x'].val,
                        current_params['resolution_y'].val,
                        current_params['enable_scv'].val,
                        current_params['write_log'].val,
                        current_params['multi_approach'].val
                    )

            self.update = getESMTracker

        elif type == 'ict':
            def getICTracker(feature, current_params=self.params):
                version = current_params['version'].val
                if version == 'python':
                    print 'Using python version'
                    return BMICTracker(
                        max_iterations=current_params['max_iterations'].val,
                        threshold=current_params['threshold'].val,
                        res=(current_params['resolution_x'].val,
                             current_params['resolution_y'].val),
                        use_scv=current_params['enable_scv'].val,
                        multi_approach=current_params['multi_approach'].val,
                        feature=feature)
                elif cython_found and version == 'cython':
                    print 'Using cython version'
                    # NNTracker(nn_iters, nn_samples, res[0], res[1], 0.06, 0.04, use_scv)
                    return BMICTracker_c(
                        current_params['max_iterations'].val,
                        current_params['threshold'].val,
                        current_params['resolution_x'].val,
                        current_params['resolution_y'].val,
                        current_params['enable_scv'].val)

            self.update = getICTracker
        elif type == 'cascade':
            def initTrackers(feature, current_params):
                trackers = []
                for i in xrange(len(current_params['trackers'].val)):
                    tracker_type = current_params['trackers'].val[i]
                    if tracker_type == 'none':
                        continue
                    tracker_params = current_params['parameters'].val[i]
                    if tracker_params is None:
                        param_list = current_params['parameters'].list
                        tracker_params = TrackingParams(tracker_type, param_list[tracker_type])
                    trackers.append(tracker_params.update(feature, tracker_params.params))
                return trackers

            def getCascadeTracker(feature, current_params=self.params):
                version = current_params['version'].val
                if version == 'python':
                    print 'Using python version'
                    return CascadeTracker(
                        initTrackers(feature, current_params), feature=feature)
                elif cython_found and version == 'cython':
                    print 'Using cython version'
                    # NNTracker(nn_iters, nn_samples, res[0], res[1], 0.06, 0.04, use_scv)
                    return CascadeTracker_c(
                        initTrackers(feature, current_params))

            self.update = getCascadeTracker
        elif type == 'l1':
            # pass
            self.update = lambda feature, current_params=self.params: L1Tracker(
                no_of_samples=current_params['no_of_samples'].val,
                angle_threshold=current_params['angle_threshold'].val,

                res=[current_params['resolution_x'].val,
                     current_params['resolution_y'].val],
                no_of_templates=current_params['no_of_templates'].val,
                alpha=current_params['alpha'].val,
                use_scv=current_params['enable_scv'].val,
                multi_approach=current_params['multi_approach'].val,
                feature=feature)
        elif xvision_found and type == 'xv_ssd':
            self.update = lambda feature, current_params=self.params: XVSSDTracker(
                steps_per_frame=current_params['steps_per_frame'].val,
                multi_approach=current_params['multi_approach'].val,
                enable_scv=current_params['enable_scv'].val,
                direct_capture=current_params['direct_capture'].val,
                stepper=current_params['stepper'].val,
                show_xv_window=current_params['show_xv_window'].val,
                use_pyramidal_stepper=current_params['use_pyramidal_stepper'].val,
                no_of_levels=current_params['no_of_levels'].val,
                scale=current_params['scale'].val)

        elif type == 'desm':
            self.update = lambda feature, current_params=self.params: DESMTracker(
                current_params['max_iterations'].val,
                current_params['threshold'].val,
                current_params['resolution_x'].val,
                current_params['resolution_y'].val,
                current_params['enable_scv'].val,
                self.mmodels[current_params['dof'].val])
        elif type == 'dnn':
            self.update = lambda feature, current_params=self.params: DNNTracker(
                current_params['max_iterations'].val,
                current_params['no_of_samples'].val,
                current_params['resolution_x'].val,
                current_params['resolution_y'].val,
                current_params['sigma_t'].val,
                current_params['sigma_d'].val,
                current_params['enable_scv'].val,
                self.mmodels[current_params['dof'].val])
        elif type == 'dlk':
            self.update = lambda feature, current_params=self.params: DLKTracker(
                current_params['max_iterations'].val,
                current_params['threshold'].val,
                current_params['resolution_x'].val,
                current_params['resolution_y'].val,
                current_params['enable_scv'].val,
                self.mmodels[current_params['dof'].val])
        elif type == 'pf':
            self.update = lambda feature, current_params=self.params: PFTracker(
                current_params['no_of_samples'].val,
                current_params['resolution_x'].val,
                current_params['resolution_y'].val,
                current_params['enable_scv'].val,
                self.mmodels[current_params['dof'].val])
        elif type == 'rkl':
            self.update = lambda feature, current_params=self.params: RKLTracker(
                current_params['max_iterations'].val,
                current_params['threshold'].val,
                current_params['resolution_x'].val,
                current_params['resolution_y'].val,
                current_params['enable_scv'].val)
        elif flann_found and type == 'nnic':
            def getNNICTracker(feature, current_params=self.params):
                version = current_params['version'].val
                trackers = []
                if version == 'python':
                    print 'Using python version'
                    trackers.append(
                        NNTracker(
                            no_of_samples=current_params['no_of_samples'].val,
                            no_of_iterations=current_params['nn_no_of_iterations'].val,
                            res=(current_params['nn_resolution_x'].val,
                                 current_params['nn_resolution_y'].val),
                            use_scv=current_params['nn_enable_scv'].val,
                            multi_approach=current_params['multi_approach'].val,
                            warp_generator=lambda: random_homography(current_params['nn_sigma_d'].val,
                                                                     current_params['nn_sigma_t'].val),
                            feature=feature,
                        )
                    )
                    trackers.append(
                        BMICTracker(
                            max_iterations=current_params['ic_max_iterations'].val,
                            threshold=current_params['ic_threshold'].val,
                            res=(current_params['ic_resolution_x'].val,
                                 current_params['ic_resolution_y'].val),
                            use_scv=current_params['ic_enable_scv'].val,
                            multi_approach=current_params['multi_approach'].val,
                            feature=feature
                        )
                    )
                    return CascadeTracker(trackers, feature=feature)
                elif cython_found and version == 'cython':
                    print 'Using cython version'
                    # NNTracker(nn_iters, nn_samples, res[0], res[1], 0.06, 0.04, use_scv)
                    trackers.append(
                        NNTracker_c(
                            current_params['nn_no_of_iterations'].val,
                            current_params['no_of_samples'].val,
                            current_params['nn_resolution_x'].val,
                            current_params['nn_resolution_y'].val,
                            current_params['nn_sigma_t'].val,
                            current_params['nn_sigma_d'].val,
                            current_params['nn_enable_scv'].val
                        )
                    )
                    trackers.append(
                        BMICTracker_c(
                            current_params['ic_max_iterations'].val,
                            current_params['ic_threshold'].val,
                            current_params['ic_resolution_x'].val,
                            current_params['ic_resolution_y'].val,
                            current_params['ic_enable_scv'].val
                        )
                    )
                    return CascadeTracker_c(trackers)

            self.update = getNNICTracker
        elif flann_found and type == 'tt_dnn_bmic':
            self.update = lambda feature, current_params=self.params: cy_tt.make_dnn_bmic(
                use_scv=current_params['enable_scv'].val,
                res=(current_params['resolution_x'].val, current_params['resolution_y'].val),
                nn_iters=current_params['nn_max_iterations'].val,
                nn_samples=current_params['no_of_samples'].val,
                bmic_iters=current_params['ic_max_iterations'].val,
                MModel=current_params['MModel'].val,
                gnn=current_params['enable_gnn'].val,
                exp=current_params['enable_exp'].val
            )
        elif flann_found and type == 'tt_nn_bmic':
            self.update = lambda feature, current_params=self.params: cy_tt.make_nn_bmic(
                use_scv=current_params['enable_scv'].val,
                res=(current_params['resolution_x'].val, current_params['resolution_y'].val),
                nn_iters=current_params['nn_max_iterations'].val,
                nn_samples=current_params['no_of_samples'].val,
                bmic_iters=current_params['ic_max_iterations'].val
            )
        elif mtf_found and type == 'mtf':
            self.update = lambda feature, current_params=self.params: mtfTracker(
                current_params['config_root_dir'].val)
        else:
            self.init_success = False
            print "Invalid tracker:", type

    def getSortedParams(self):
        return sorted(self.params.values(), key=lambda k: k.id)


    def printParamValue(self):
        print 'Initialized', self.type.capitalize(), 'tracker with:'
        # print self.type, "filter:"
        for key in self.params.keys():
            print key, "=", self.params[key].val


class Param:
    def __init__(self, name, id, val, type, list=None):
        self.id = id
        self.name = name
        self.val = val
        self.type = type
        self.list = list



