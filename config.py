import numpy as np

from Misc import getParamDict


def getBasicParams(trackers, xv_input_found, extended_db):
    video_pipeline = ['OpenCV']
    cv_sources = ['jpeg', 'mpeg', 'usb camera']
    sources = [cv_sources]
    if xv_input_found:
        video_pipeline.append('XVision')
        xv_sources = ['mpeg', 'usb camera', 'firewire camera']
        sources.append(xv_sources)
    initialization = ['manual', 'ground_truth']
    color_spaces = ['Grayscale', 'RGB', 'HSV', 'YCrCb', 'HLS', 'Lab']
    filters = ['none', 'gabor', 'laplacian', 'sobel', 'scharr', 'canny', 'LoG', 'DoG']
    features = ['none', 'hoc']
    smoothing = ['none', 'box', 'bilateral', 'gauss', 'median']
    smoothing_kernel = map(str, range(3, 26, 2))
    params = [video_pipeline, sources, initialization, color_spaces, filters, features, smoothing, smoothing_kernel,
              trackers]
    labels = ['pipeline', 'source', 'initialization', 'color_space', 'filter', 'feature', 'smoothing',
              'smoothing_kernel', 'tracker']
    default_id = [video_pipeline.index('OpenCV'),
                  sources[video_pipeline.index('OpenCV')].index('jpeg'),
                  initialization.index('ground_truth'),
                  color_spaces.index('Grayscale'),
                  filters.index('none'),
                  features.index('none'),
                  smoothing.index('gauss'),
                  smoothing_kernel.index(str(5)),
                  trackers.index('rkl')
    ]

    if extended_db:
        params_dict = getParamDict()
        task_type = [params_dict['actors'][id] for id in xrange(len(params_dict['actors']))]
        tasks = [params_dict['sequences'][actor] for actor in task_type]
        tasks = [seq.values() for seq in tasks]
        params.extend([task_type, tasks])
        labels.extend(['type', 'task'])
        default_id.extend([0, 3])
    else:
        task_type = ['simple', 'complex']
        light_conditions = ['nl', 'dl']
        speeds = ['s1', 's2', 's3', 's4', 's5', 'si']
        complex_tasks = ['bus', 'highlighting', 'letter', 'newspaper']
        simple_tasks = ['bookI', 'bookII', 'bookIII', 'cereal', 'juice', 'mugI', 'mugII', 'mugIII']
        robot_tasks = ['robot_bookI', 'robot_bookII', 'robot_bookIII', 'robot_cereal', 'robot_juice', 'robot_mugI',
                       'robot_mugII', 'robot_mugIII']
        tasks = [simple_tasks, complex_tasks, robot_tasks]
        params.extend([task_type, light_conditions, speeds, tasks])
        labels.extend(['type', 'light', 'speed', 'task'])
        default_id.extend([
            task_type.index('simple'),
            light_conditions.index('nl'),
            speeds.index('s3'),
            tasks[task_type.index('simple')].index('juice')
        ])
    return params, labels, default_id


def getTrackingParams(flann_found, cython_found,
                      xvision_found, mtf_found,
                      simple_only=False):
    multichannel_nn = ['none', 'mean', 'majority', 'flatten']
    multichannel_esm = ['none', 'mean', 'flatten']
    multichannel_ict = ['none', 'mean', 'flatten']
    multichannel_l1 = ['none', 'mean', 'flatten']
    multichannel_generic = ['none']

    params_nn = {
        'no_of_samples': {'id': 0, 'default': 500, 'type': 'int', 'list': range(100, 5000, 100)},
        'no_of_iterations': {'id': 1, 'default': 2, 'type': 'int', 'list': range(1, 100, 1)},
        'resolution_x': {'id': 2, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'resolution_y': {'id': 3, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'multi_approach': {'id': 5, 'default': 'none', 'type': 'string', 'list': multichannel_nn},
        'sigma_t': {'id': 1, 'default': 0.06, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'sigma_d': {'id': 1, 'default': 0.04, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'version': {'id': 6, 'default': 'cython', 'type': 'string', 'list': ['python', 'cython']}
    }
    params_esm = {
        'max_iterations': {'id': 0, 'default': 30, 'type': 'int', 'list': range(1, 100, 1)},
        'threshold': {'id': 1, 'default': 0.0001, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'err_thresh': {'id': 1, 'default': 10, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'resolution_x': {'id': 2, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'resolution_y': {'id': 3, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'enable_err_thresh': {'id': 4, 'default': False, 'type': 'boolean', 'list': [True, False]},
        'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'multi_approach': {'id': 5, 'default': 'flatten', 'type': 'string', 'list': multichannel_esm},
        'version': {'id': 6, 'default': 'cython', 'type': 'string', 'list': ['python', 'cython']},
        'write_log': {'id': 7, 'default': False, 'type': 'boolean', 'list': [True, False]}
    }
    params_ict = {
        'max_iterations': {'id': 0, 'default': 30, 'type': 'int', 'list': range(1, 100, 1)},
        'threshold': {'id': 1, 'default': 0.0001, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'resolution_x': {'id': 2, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'resolution_y': {'id': 3, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'multi_approach': {'id': 5, 'default': 'flatten', 'type': 'string', 'list': multichannel_ict},
        'version': {'id': 6, 'default': 'cython', 'type': 'string', 'list': ['python', 'cython']}
    }
    params_l1 = {
        'no_of_samples': {'id': 0, 'default': 10, 'type': 'int', 'list': range(10, 500, 10)},
        'angle_threshold': {'id': 1, 'default': 2, 'type': 'int', 'list': np.arange(1, 10, 0.5)},
        'resolution_x': {'id': 2, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
        'resolution_y': {'id': 3, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
        'no_of_templates': {'id': 4, 'default': 10, 'type': 'int', 'list': range(5, 100, 5)},
        'alpha': {'id': 5, 'default': 50, 'type': 'float', 'list': range(100, 5000, 100)},
        'enable_scv': {'id': 6, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'multi_approach': {'id': 7, 'default': 'flatten', 'type': 'string', 'list': multichannel_l1},
        'version': {'id': 8, 'default': 'python', 'type': 'string', 'list': ['python', 'cython']}
    }
    params_xv_ssd = {
        'steps_per_frame': {'id': 0, 'default': 10, 'type': 'int', 'list': range(10, 500, 10)},
        'multi_approach': {'id': 1, 'default': 'mean', 'type': 'string', 'list': multichannel_generic},
        'stepper': {'id': 2, 'default': 'trans', 'type': 'string',
                    'list': ['trans', 'rotate', 'rt', 'se2']},
        'use_pyramidal_stepper': {'id': 3, 'default': False, 'type': 'boolean', 'list': [True, False]},
        'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'direct_capture': {'id': 5, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'show_xv_window': {'id': 6, 'default': False, 'type': 'boolean', 'list': [True, False]},
        'no_of_levels': {'id': 7, 'default': 2, 'type': 'int', 'list': range(1, 100, 1)},
        'scale': {'id': 8, 'default': 0.5, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)}
    }

    params_desm = {
        'max_iterations': {'id': 0, 'default': 10, 'type': 'int', 'list': range(1, 100, 1)},
        'threshold': {'id': 1, 'default': 0.01, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'resolution_x': {'id': 2, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'resolution_y': {'id': 3, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'enable_scv': {'id': 4, 'default': False, 'type': 'boolean', 'list': [True, False]},
        'multi_approach': {'id': 5, 'default': 'flatten', 'type': 'string', 'list': multichannel_generic},
        'dof': {'id': 7, 'default': 8, 'type': 'discrete', 'list': [2, 4, 6, 8]}
    }
    params_dnn = {
        'no_of_samples': {'id': 0, 'default': 500, 'type': 'int', 'list': range(100, 5000, 100)},
        'max_iterations': {'id': 1, 'default': 2, 'type': 'int', 'list': range(1, 100, 1)},
        'resolution_x': {'id': 2, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'resolution_y': {'id': 3, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'multi_approach': {'id': 5, 'default': 'none', 'type': 'string', 'list': multichannel_generic},
        'sigma_t': {'id': 6, 'default': 0.06, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'sigma_d': {'id': 7, 'default': 0.04, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'dof': {'id': 8, 'default': 8, 'type': 'discrete', 'list': [2, 3, 4, 6, 8]}
    }

    params_dlk = {
        'max_iterations': {'id': 0, 'default': 30, 'type': 'int', 'list': range(1, 100, 1)},
        'threshold': {'id': 1, 'default': 0.01, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'resolution_x': {'id': 2, 'default': 100, 'type': 'int', 'list': range(5, 100, 5)},
        'resolution_y': {'id': 3, 'default': 100, 'type': 'int', 'list': range(5, 100, 5)},
        'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'multi_approach': {'id': 5, 'default': 'flatten', 'type': 'string', 'list': multichannel_generic},
        'dof': {'id': 6, 'default': 8, 'type': 'discrete', 'list': [2, 3, 4, 6, 8]}
    }

    params_pf = {
        'no_of_samples': {'id': 0, 'default': 500, 'type': 'int', 'list': range(100, 5000, 100)},
        'resolution_x': {'id': 2, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'resolution_y': {'id': 3, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'multi_approach': {'id': 5, 'default': 'none', 'type': 'string', 'list': multichannel_generic},
        'dof': {'id': 6, 'default': 8, 'type': 'discrete', 'list': [2, 4, 6, 8]}
    }

    params_rkl = {
        'max_iterations': {'id': 0, 'default': 100, 'type': 'int', 'list': range(1, 100, 1)},
        'threshold': {'id': 1, 'default': 0.001, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'resolution_x': {'id': 2, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
        'resolution_y': {'id': 3, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
        'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'multi_approach': {'id': 5, 'default': 'flatten', 'type': 'string', 'list': multichannel_generic}
    }

    params_nnic = {
        'no_of_samples': {'id': 0, 'default': 1000, 'type': 'int', 'list': range(100, 5000, 100)},
        'nn_no_of_iterations': {'id': 1, 'default': 10, 'type': 'int', 'list': range(1, 100, 1)},
        'nn_resolution_x': {'id': 2, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'nn_resolution_y': {'id': 3, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'nn_enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'nn_sigma_t': {'id': 5, 'default': 0.06, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'nn_sigma_d': {'id': 6, 'default': 0.04, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'multi_approach': {'id': 7, 'default': 'none', 'type': 'string', 'list': multichannel_nn},
        'ic_max_iterations': {'id': 8, 'default': 5, 'type': 'int', 'list': range(1, 100, 1)},
        'ic_threshold': {'id': 9, 'default': 0.0001, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
        'ic_resolution_x': {'id': 10, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'ic_resolution_y': {'id': 11, 'default': 50, 'type': 'int', 'list': range(5, 100, 5)},
        'ic_enable_scv': {'id': 12, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'version': {'id': 13, 'default': 'cython', 'type': 'string', 'list': ['python', 'cython']}
    }
    params_mtf = {
        'config_root_dir': {'id': 0, 'default': 'C++/MTF/Config', 'type': 'string', 'list': None},
        'multi_approach': {'id': 5, 'default': 'flatten', 'type': 'string', 'list': multichannel_generic}
    }

    params_tt_nn_bmic = {
        'no_of_samples': {'id': 0, 'default': 1100, 'type': 'int', 'list': range(100, 10000, 100)},
        'nn_max_iterations': {'id': 0, 'default': 10, 'type': 'int', 'list': range(1, 100, 1)},
        'ic_max_iterations': {'id': 0, 'default': 5, 'type': 'int', 'list': range(1, 100, 1)},
        'resolution_x': {'id': 2, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
        'resolution_y': {'id': 2, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
        'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'multi_approach': {'id': 0, 'default': 'flatten', 'type': 'string', 'list': multichannel_generic}
    }
    params_tt_dnn_bmic = {
        'no_of_samples': {'id': 0, 'default': 1100, 'type': 'int', 'list': range(100, 10000, 100)},
        'nn_max_iterations': {'id': 0, 'default': 10, 'type': 'int', 'list': range(1, 100, 1)},
        'ic_max_iterations': {'id': 0, 'default': 5, 'type': 'int', 'list': range(1, 100, 1)},
        'resolution_x': {'id': 2, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
        'resolution_y': {'id': 2, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
        'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
        'multi_approach': {'id': 0, 'default': 'flatten', 'type': 'string', 'list': multichannel_generic},
        'MModel': {'id': 6, 'default': 1, 'type': 'discrete', 'list': [1, 2, 3, 4, 5, 6]},
        'enable_gnn': {'id': 4, 'default': False, 'type': 'boolean', 'list': [True, False]},
        'enable_exp': {'id': 4, 'default': False, 'type': 'boolean', 'list': [True, False]},
    }
    tracking_params = {'esm': params_esm, 'ict': params_ict, 'l1': params_l1,
                       'rkl': params_rkl, 'pf': params_pf}

    if flann_found:
        tracking_params['nn'] = params_nn
        tracking_params['tt_nn_bmic'] = params_tt_nn_bmic
        tracking_params['nnic'] = params_nnic
        if cython_found:
            tracking_params['dnn'] = params_dnn
            tracking_params['tt_dnn_bmic'] = params_tt_dnn_bmic
    if cython_found:
        tracking_params['desm'] = params_desm
        tracking_params['dlk'] = params_dlk
    if xvision_found:
        tracking_params['xv_ssd'] = params_xv_ssd
    if mtf_found:
        tracking_params['mtf'] = params_mtf

    if not simple_only:
        tracker_list = tracking_params.keys()
        params_cascade = {'trackers': {'id': 1, 'default': ['nn', 'ict', 'none', 'none', 'none'],
                                       'type': 'string_list', 'list': tracker_list},
                          'parameters': {'id': 2, 'default': [None, None, None, None, None],
                                         'type': 'tracking_params', 'list': getTrackingParams(flann_found, cython_found,
                                                                                              xvision_found, mtf_found,
                                                                                              simple_only=True)},
                          'version': {'id': 0, 'default': 'python', 'type': 'string', 'list': ['python', 'cython']}
        }
        tracking_params['cascade'] = params_cascade
    return tracking_params


def getFilteringParams():
    params_none = {}
    params_gabor = {'ksize': {'id': 0, 'default': {'base': 2, 'mult': 1, 'limit': 10, 'add': 1}, 'type': 'int'},
                    'sigma': {'id': 1, 'default': {'base': 0.1, 'mult': 10, 'limit': 100, 'add': 0.0}, 'type': 'float'},
                    'theta': {'id': 2, 'default': {'base': np.pi / 12, 'mult': 0, 'limit': 24, 'add': 0.0},
                              'type': 'float'},
                    'lambd': {'id': 3, 'default': {'base': 0.1, 'mult': 10, 'limit': 100, 'add': 10.0},
                              'type': 'float'},
                    'gamma': {'id': 4, 'default': {'base': 0.1, 'mult': 10, 'limit': 100, 'add': 0.0}, 'type': 'float'}
    }
    params_laplacian = {'ksize': {'id': 0, 'default': {'base': 2, 'mult': 1, 'limit': 10, 'add': 1}, 'type': 'int'},
                        'scale': {'id': 1, 'default': {'base': 1, 'mult': 0, 'limit': 10, 'add': 1}, 'type': 'int'},
                        'delta': {'id': 2, 'default': {'base': 1, 'mult': 0, 'limit': 255, 'add': 0}, 'type': 'int'},
    }
    params_sobel = {'ksize': {'id': 0, 'default': {'base': 2, 'mult': 1, 'limit': 10, 'add': 1}, 'type': 'int'},
                    'scale': {'id': 1, 'default': {'base': 1, 'mult': 0, 'limit': 10, 'add': 1}, 'type': 'int'},
                    'delta': {'id': 2, 'default': {'base': 1, 'mult': 0, 'limit': 255, 'add': 0}, 'type': 'int'},
                    'dx': {'id': 3, 'default': {'base': 1, 'mult': 1, 'limit': 5, 'add': 0}, 'type': 'int'},
                    'dy': {'id': 4, 'default': {'base': 1, 'mult': 0, 'limit': 5, 'add': 0}, 'type': 'int'}
    }
    params_scharr = {'scale': {'id': 0, 'default': {'base': 1, 'mult': 0, 'limit': 10, 'add': 1}, 'type': 'int'},
                     'delta': {'id': 1, 'default': {'base': 1, 'mult': 0, 'limit': 255, 'add': 0}, 'type': 'int'},
                     'dx': {'id': 2, 'default': {'base': 1, 'mult': 1, 'limit': 1, 'add': 0}, 'type': 'int'},
                     'dy': {'id': 3, 'default': {'base': 1, 'mult': 0, 'limit': 1, 'add': 0}, 'type': 'int'},
    }
    params_canny = {'low_thresh': {'id': 0, 'default': {'base': 1, 'mult': 20, 'limit': 50, 'add': 0}, 'type': 'int'},
                    'ratio': {'id': 1, 'default': {'base': 1, 'mult': 4, 'limit': 10, 'add': 0}, 'type': 'float'},
    }
    params_dog = {'ksize': {'id': 0, 'default': {'base': 2, 'mult': 1, 'limit': 10, 'add': 1}, 'type': 'int'},
                  'exc_std': {'id': 1, 'default': {'base': 0.1, 'mult': 20, 'limit': 100, 'add': 1}, 'type': 'float'},
                  'inh_std': {'id': 2, 'default': {'base': 0.1, 'mult': 28, 'limit': 100, 'add': 1}, 'type': 'float'},
                  'ratio': {'id': 3, 'default': {'base': 0.05, 'mult': 50, 'limit': 200, 'add': 0.0}, 'type': 'float'},
    }
    params_log = {'gauss_ksize': {'id': 0, 'default': {'base': 2, 'mult': 1, 'limit': 10, 'add': 1}, 'type': 'int'},
                  'std': {'id': 1, 'default': {'base': 0.1, 'mult': 20, 'limit': 100, 'add': 0.1}, 'type': 'float'},
                  'lap_ksize': {'id': 2, 'default': {'base': 2, 'mult': 1, 'limit': 5, 'add': 1}, 'type': 'int'},
                  'scale': {'id': 3, 'default': {'base': 1, 'mult': 0, 'limit': 10, 'add': 1}, 'type': 'int'},
                  'delta': {'id': 4, 'default': {'base': 1, 'mult': 0, 'limit': 255, 'add': 0}, 'type': 'int'}
    }
    filtering_params = {'none': params_none, 'gabor': params_gabor, 'laplacian': params_laplacian,
                        'sobel': params_sobel, 'scharr': params_scharr, 'canny': params_canny,
                        'DoG': params_dog, 'LoG': params_log}
    return filtering_params

