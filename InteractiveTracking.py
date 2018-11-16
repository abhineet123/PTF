import os
import time
from TrackingParams import TrackingParams
from FilteringParams import *
from GUI import *
from Misc import *
from ImageUtils import *
from matplotlib import pyplot as plt

xv_input_found = True
try:
    import CModules.xvInput as xvInput
except:
    xv_input_found = False


class InteractiveTrackingApp:
    def __init__(self, init_frame, root_path, track_window_name, params,
                 tracking_params, filtering_params, labels, default_id=None,
                 success_threshold=5, batch_mode=False, agg_filename=None,
                 avg_filename=None, anim_app=None, extended_db=False,
                 write_tracking_data=False, tracking_data_fname=None, camera_id = 0):
        self.auto_start = False
        self.extended_db = extended_db
        self.root_path = root_path
        self.params = params
        self.agg_filename = agg_filename
        self.avg_filename = avg_filename
        self.anim_app = anim_app
        # self.tracking_params=tracking_params
        # self.filtering_params=filtering_params
        self.labels = labels
        if default_id is None:
            default_id = [0 for i in xrange(len(self.params))]
        self.default_id = default_id
        self.first_call = True

        if len(self.default_id) != len(self.params):
            raise SyntaxError('Mismatch between the sizes of default ids and params')

        if len(self.labels) != len(self.params):
            raise SyntaxError('Mismatch between the sizes of labels and params')

        # initialize trackers
        tracker_index = labels.index('tracker')
        # self.tracker_ids=dict(zip(params[tracker_index], [i for i in xrange(len(params[tracker_index]))]))
        self.tracker_type = params[tracker_index][default_id[tracker_index]]
        self.trackers = {}
        for tracker_type in params[tracker_index]:
            self.trackers[tracker_type] = TrackingParams(tracker_type, tracking_params[tracker_type])
        self.tracker = None

        # initialize filters
        filter_index = labels.index('filter')
        # self.filters_ids=dict(zip(params[filter_index], [i for i in xrange(len(params[filter_index]))]))
        self.filter_type = params[filter_index][default_id[filter_index]]
        self.filters = {}
        for filter_type in params[filter_index]:
            self.filters[filter_type] = FilteringParams(filter_type, filtering_params[filter_type])

        self.source = default_id[labels.index('source')]

        self.init_frame = init_frame
        self.track_window_name = track_window_name
        self.proc_window_name = 'Processed Images'
        self.count = 0

        self.actor = None

        self.gray_img = None
        self.proc_img = None
        self.paused = False

        self.smooth_image = None
        self.smoothing_type = None
        self.smoothing_kernel = None

        self.start_time = 0
        self.start_time_win = 0
        self.last_time = 0
        self.current_time = 0
        self.average_fps = 0
        self.average_fps_win = 0
        self.current_fps_win = 0
        self.current_fps = 0
        self.plot_fps = True

        self.curr_fps_list = []
        self.curr_fps_win_list = []
        self.avg_fps_list = []
        self.curr_error_list = []

        self.window_inited = False
        self.init_track_window = True
        self.init_params = []
        self.times = 1
        self.max_cam = 3

        self.reset = False
        self.exit_event = False
        self.cap = None
        self.from_cam = False
        self.src_img = None
        self.img = None
        self.no_of_frames = 1
        self.data_file = None
        self.camera_id = camera_id

        self.updates = None
        self.success_threshold = success_threshold

        self.initPlotParams()
        self.tracker_pause = False

        self.batch_mode = batch_mode
        self.inited = False

        self.success_count = 0
        self.success_drift = []

        self.last_update = None
        self.current_update = None
        self.last_corners = None
        self.current_corners = None

        self.write_tracking_data = write_tracking_data
        self.tracking_data_fname = tracking_data_fname
        self.tracking_data_root_dir = 'Results'
        self.tracking_data_dir = None
        self.tracking_data_path = None
        self.tracking_data_file = None

        if self.batch_mode:
            init_params = self.getInitParams()
            self.initSystem(init_params)
        else:
            gui_title = "Basic Parameters"
            self.gui_obj = GUI(self, gui_title)
            # self.gui_obj.initGUI()
            # self.gui_obj.root.mainloop()

    def getInitParams(self):
        init_params = []
        for i in xrange(len(self.params)):
            if self.labels[i] == 'task':
                type_index = self.labels.index('type')
                param = self.params[i][self.default_id[type_index]][self.default_id[i]]
            elif self.labels[i] == 'source':
                pipeline_index = self.labels.index('pipeline')
                param = self.params[i][self.default_id[pipeline_index]][self.default_id[i]]
            else:
                param = self.params[i][self.default_id[i]]
            init_params.append(param)
        # print 'init_params=', init_params
        # sys.exit()
        return init_params

    def processDatasetParams(self):
        if self.extended_db:
            actor = self.init_params[self.labels.index('type')]
            self.data_file = self.init_params[self.labels.index('task')]
        else:
            actor = 'TMT'
            type = self.init_params[self.labels.index('type')]
            light = self.init_params[self.labels.index('light')]
            speed = self.init_params[self.labels.index('speed')]
            task = self.init_params[self.labels.index('task')]
            if type == 'simple':
                self.data_file = light + '_' + task + '_' + speed
            elif type == 'complex':
                self.data_file = light + '_' + task
            else:
                print "Invalid task type specified: %s" % type
                return False

        self.actor = actor
        self.dataset_path = self.root_path + '/' + actor
        print "Getting input from data: ", self.data_file

        self.img_path = self.dataset_path + '/' + self.data_file

        if self.init_method == 'ground_truth':
            self.ground_truth = readTrackingData(self.img_path + '.txt')
            if self.ground_truth is None:
                print 'Using manual initialization instead...'
                self.init_method = 'manual'
                return
            # self.updates=getGroundTruthUpdates(self.dataset_path + '/' + data_file + '.txt')
            self.no_of_frames = self.ground_truth.shape[0]
            print "no_of_frames=", self.no_of_frames
            self.initparam = [self.ground_truth[self.init_frame, 0:2].tolist(),
                              self.ground_truth[self.init_frame, 2:4].tolist(),
                              self.ground_truth[self.init_frame, 4:6].tolist(),
                              self.ground_truth[self.init_frame, 6:8].tolist()]
            # print tracking_data
            print "object location initialized to:", self.initparam


    def initSystem(self, init_params):
        print "\n" + "*" * 60 + "\n"

        # if not self.first_call:
        # self.tracker.cleanup()

        self.inited = False
        self.success_count = 0
        self.success_drift = []

        self.init_params = init_params

        self.init_method = self.init_params[self.labels.index('initialization')]
        self.pipeline = self.init_params[self.labels.index('pipeline')]
        self.source = self.init_params[self.labels.index('source')]
        self.feature = self.init_params[self.labels.index('feature')]

        self.color_space = self.init_params[self.labels.index('color_space')]
        # multi_approach=self.trackers[self.tracker_type].params['multi_approach'].val

        if self.color_space.lower() != 'grayscale':
            self.multi_channel = True
        else:
            self.multi_channel = False

        self.tracker_type = self.init_params[self.labels.index('tracker')]
        if not self.multi_channel:
            print 'Disabling multichannel'
            try:
                self.trackers[self.tracker_type].params['multi_approach'].val = 'none'
            except KeyError:
                for sub_tracker in self.trackers[self.tracker_type].params['parameters'].val:
                    if sub_tracker is None:
                        continue
                    sub_tracker.params['multi_approach'].val = 'none'
                    print '\n\n Here we are \n\n'
        self.tracker = self.trackers[self.tracker_type].update(self.feature, self.trackers[self.tracker_type].params)

        self.smoothing_type = self.init_params[self.labels.index('smoothing')]
        self.smoothing_kernel = int(self.init_params[self.labels.index('smoothing_kernel')])
        if self.smoothing_type == 'none':
            print 'Smoothing is disabled'
            self.smooth_image = lambda src: src
        else:
            print 'Smoothing images using ' + self.smoothing_type + ' filter with kernel size ', self.smoothing_kernel
            if self.smoothing_type == 'box':
                self.smooth_image = lambda src: cv2.blur(src, (self.smoothing_kernel, self.smoothing_kernel))
            elif self.smoothing_type == 'bilateral':
                self.smooth_image = lambda src: cv2.bilateralFilter(src, self.smoothing_kernel, 100, 100)
            elif self.smoothing_type == 'gauss':
                self.smooth_image = lambda src: cv2.GaussianBlur(src, (self.smoothing_kernel, self.smoothing_kernel), 3)
            elif self.smoothing_type == 'median':
                self.smooth_image = lambda src: cv2.medianBlur(src, self.smoothing_kernel)

        old_filter_type = self.filter_type
        self.filter_type = self.init_params[self.labels.index('filter')]
        if not self.batch_mode and old_filter_type != self.filter_type:
            self.initFilterWindow()

        if self.filter_type == 'none':
            print "Filtering disabled"
        elif self.filter_type in self.filters.keys():
            try:
                self.tracker.use_scv = False
            except AttributeError:
                for tracker in self.tracker.trackers:
                    tracker.use_scv = False
            print "Using %s filtering" % self.filter_type
        else:
            print 'Invalid filter type: ', self.filter_type
            return False

        # print "Using ", self.tracker_name, " tracker"
        if self.source == 'jpeg' or self.source == 'mpeg':
            self.processDatasetParams()
        else:
            self.data_file=self.source

        if xv_input_found and self.pipeline == 'XVision':
            if self.source == 'usb camera':
                self.from_cam = True
                [width, height] = xvInput.initSource(3, None, None)
            elif self.source == 'firewire camera':
                self.from_cam = True
                [width, height] = xvInput.initSource(4, None, None)
            elif self.source == 'mpeg':
                mpeg_fname = self.img_path + '.mpg'
                [width, height] = xvInput.initSource(1, mpeg_fname, None)
            elif self.source == 'avi':
                avi_fname = self.img_path + '.avi'
                [width, height] = xvInput.initSource(2, avi_fname, None)
            else:
                raise SystemExit('Invalid XVision source specified')
            self.src_img = np.zeros((height, width, 3)).astype(np.uint8)
        elif self.pipeline == 'OpenCV':
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture()
            if self.source == 'usb camera':
                print "Initializing camera..."
                self.from_cam = True
                if not self.cap.open(self.camera_id):
                    raise SystemExit("No valid camera found")
                dWidth = self.cap.get(3)
                dHeight = self.cap.get(4)
                if dWidth == 0 or dHeight == 0:
                    raise SystemExit("No valid camera found")
                print "Frame size : ", dWidth, " x ", dHeight
            elif self.source == 'mpeg':
                mpeg_fname = self.img_path + '.mpg'
                if not self.cap.open(mpeg_fname):
                    print 'MPEG file ', mpeg_fname, 'could not be opened'
                    sys.exit()
            elif self.source == 'jpeg':
                jpeg_fname = self.img_path + '/frame%05d.jpg'
                if not self.cap.open(jpeg_fname):
                    print 'JPEG files ', jpeg_fname, 'could not be accessed'
                    sys.exit()
            else:
                raise SystemExit('Invalid OpenCV source specified')
        else:
            raise StandardError('Invalid video pipeline specified')

        if self.write_tracking_data:
            if self.actor is not None:
                self.tracking_data_dir = self.tracking_data_root_dir + '/' + self.actor + '/' + self.data_file
            else:
                self.tracking_data_dir = self.tracking_data_root_dir + '/' + self.data_file
            if not os.path.exists(self.tracking_data_dir):
                print 'Creating tracking data folder: {:s}'.format(self.tracking_data_dir)
                os.makedirs(self.tracking_data_dir)
            if self.from_cam:
                self.plot_fps = True
                self.tracking_data_path = self.tracking_data_dir + '/camera_res_{:s}.txt'.format(self.tracker_type)
            else:
                self.plot_fps = False
                if self.tracking_data_fname is None:
                    self.tracking_data_fname = '{:s}_8_0'.format(self.tracker_type)
                self.tracking_data_path = '{:s}/{:s}.txt'.format(self.tracking_data_dir, self.tracking_data_fname)
            self.tracking_data_file = open(self.tracking_data_path, 'w')
            print 'Writing tracking data to: {:s}'.format(self.tracking_data_path)
            self.tracking_data_file.write('{:>8s}{:>8s}{:>8s}{:>8s}{:>8s}{:>8s}{:>8s}{:>8s}{:>8s}\n'.format(
                'frame', 'ulx', 'uly', 'urx', 'ury', 'lrx', 'lry', 'llx', 'lly'))

        # if not self.first_call:
        # self.writeResults()
        # self.anim_app.start_anim=True

        print "\n" + "*" * 60 + "\n"
        return True

    def initPlotParams(self):

        self.curr_error = 0
        self.avg_error = 0
        self.avg_error_list = []
        self.curr_fps_list = []
        self.curr_fps_win_list = []
        self.avg_fps_list = []
        self.curr_error_list = []
        self.frame_times = []
        self.update_diff = []
        self.max_error = 0
        self.max_fps = 0
        self.max_val = 0
        self.call_count = 0

        self.count = 0
        self.current_fps_win = 0
        self.current_fps = 0
        self.average_fps = 0

        # self.start_time=datetime.now().time()
        self.start_time = 0
        self.current_time = 0
        self.last_time = 0

        self.switch_plot = True

    def on_frame(self, img, numtimes):
        # print "frame: ", numtimes
        if self.first_call and not self.batch_mode:
            # self.gui_obj.initWidgets(start_label='Restart')
            self.first_call = False

        self.count += 1
        self.times = numtimes

        self.img = self.smooth_image(img)
        # print "img.shape=",img.shape

        if self.color_space == 'RGB':
            self.proc_img = self.img
        elif self.color_space == 'Grayscale':
            self.proc_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        elif self.color_space == 'HSV':
            self.proc_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        elif self.color_space == 'YCrCb':
            self.proc_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2YCR_CB)
        elif self.color_space == 'HLS':
            self.proc_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
        elif self.color_space == 'Lab':
            self.proc_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)
        else:
            raise SystemExit('Error in on_frame:\n'
                             'Invalid color space specified:\t', self.color_space)
        # if len(self.proc_img.shape)>2:
        # for i in xrange(self.proc_img.shape[2]):
        # np.savetxt(self.color_space+'_'+str(i)+'.txt', self.proc_img[:, :, i], fmt='%12.6f', delimiter='\t')
        # else:
        # np.savetxt(self.color_space+'.txt', self.proc_img, fmt='%12.6f', delimiter='\t')

        self.proc_img = self.filters[self.filter_type].apply(self.proc_img)

        if not self.batch_mode:
            # print 'Processing frame', self.times+1, ' avg_fps:', self.average_fps,\
            # 'avg_error:', self.avg_error
            cv2.imshow(self.proc_window_name, self.proc_img)

        if self.count == 100 or self.times == self.no_of_frames - 1:
            print 'Processing frame', self.times + 1, ' avg_fps:', self.average_fps, \
                'avg_error:', self.avg_error
            self.count = 0

        self.proc_img = self.proc_img.astype(np.float64)

        write_header = 0

        self.start_time = time.clock()
        if not self.inited:
            print '\n\n initializing....\n\n'
            if not self.batch_mode:
                cv2.namedWindow(self.track_window_name)
            if self.from_cam or self.init_method == 'manual':
                self.initparam = getTrackingObject2(self.img)
                # self.initparam = np.asarray(sel_pts).astype(np.float64).T
                if len(self.initparam) < 4:
                    print 'Invalid initparam: ', self.initparam
                    self.exit_event = True
                    sys.exit()
            init_array = np.array(self.initparam, dtype=np.float64).T
            # print 'calling self.tracker.initialize....'
            self.tracker.initialize(self.proc_img, init_array)
            # print 'Done'
            self.inited = True
            write_header = 1
            # self.start_time = time.clock()
            # self.current_time = self.start_time
            # self.last_time = self.start_time
        # self.tracker.update(self.proc_img, use_update=self.updates[self.times])
        else:
            # print 'calling self.tracker.update....'
            self.tracker.update(self.proc_img)
        self.end_time = time.clock()
        # self.last_time = self.current_time
        # self.current_time = end_time

        if self.current_corners is not None:
            self.last_corners = self.current_corners.copy()
        self.current_corners = self.tracker.get_region()
        # print 'current_corners: ', self.current_corners
        if self.last_corners is None:
            self.last_corners = self.current_corners.copy()

        if self.current_update is not None:
            self.last_update = np.copy(self.current_update)
        self.current_update = compute_homography(self.last_corners, self.current_corners)
        if self.last_update is None:
            self.last_update = np.copy(self.current_update)

        diff = math.sqrt(np.sum(np.square(self.last_update - self.current_update)) / 8)
        self.update_diff.append(diff)

        if not self.from_cam and self.init_method == 'ground_truth':
            self.actual_corners = [self.ground_truth[self.times, 0:2].tolist(),
                                   self.ground_truth[self.times, 2:4].tolist(),
                                   self.ground_truth[self.times, 4:6].tolist(),
                                   self.ground_truth[self.times, 6:8].tolist()]
            self.actual_corners = np.array(self.actual_corners).T
            self.curr_error = math.sqrt(np.sum(np.square(self.actual_corners - self.current_corners)) / 4)
        else:
            self.actual_corners = self.current_corners.copy()
            self.curr_error = 0

        if self.write_tracking_data:
            self.tracking_data_file.write('%-15s%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f\n' % (
                'frame' + ('%05d' % (self.times + 1)) + '.jpg', self.current_corners[0, 0],
                self.current_corners[1, 0], self.current_corners[0, 1], self.current_corners[1, 1],
                self.current_corners[0, 2], self.current_corners[1, 2], self.current_corners[0, 3],
                self.current_corners[1, 3]))
        # writeCorners2(self.gt_corners_fname, self.current_corners, self.times + 1, write_header)

        if math.isnan(self.curr_error) or math.isinf(self.curr_error):
            print 'actual_corners:\n', self.actual_corners
            print 'tracked_corners:\n', self.current_corners
            raise SystemExit('Error in updateError:\t'
                             'Encountered invalid tracking error in frame %d' % (self.times + 1))
        if self.curr_error <= self.success_threshold:
            self.success_count += 1
            self.success_drift.append(self.curr_error)

        if self.tracker_pause:
            raw_input("Press Enter to continue...")

        return True

    def display(self):
        annotated_img = self.img.copy()
        if self.tracker.is_initialized():
            draw_region(annotated_img, self.current_corners, (0, 0, 255), 2)
            draw_region(annotated_img, self.actual_corners, (0, 255, 0), 2)
        # fps_text = "%5.2f" % self.average_fps + "   %5.2f" % self.current_fps
        fps_text = 'frame: {:4d} c:{:9.4f} a:{:9.4f} cw:{:9.4f} aw:{:9.4f} e:{:9.4f}'.format(self.times,
                                                                                             self.current_fps,
                                                                                             self.average_fps,
                                                                                             self.current_fps_win,
                                                                                             self.average_fps_win,
                                                                                             self.curr_error)
        cv2.putText(annotated_img, fps_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255))
        cv2.imshow(self.track_window_name, annotated_img)

    def initFilterWindow(self):
        if self.window_inited:
            cv2.destroyWindow(self.proc_window_name)
            self.window_inited = False
        cv2.namedWindow(self.proc_window_name, flags=cv2.CV_WINDOW_AUTOSIZE)
        if self.filter_type != 'none':
            for param in self.filters[self.filter_type].sorted_params:
                cv2.createTrackbar(param.name, self.proc_window_name,
                                   param.multiplier,
                                   param.limit, self.updateFilteringParams)
        self.window_inited = True

    def updateFilteringParams(self, val):
        if self.filters[self.filter_type].validated:
            return
        # print 'starting updateFilteringParams'
        for param in self.filters[self.filter_type].params.values():
            new_val = cv2.getTrackbarPos(param.name, self.proc_window_name)
            old_val = param.multiplier
            if new_val != old_val:
                param.updateValue(new_val)
                if not self.filters[self.filter_type].validate():
                    param.updateValue(old_val)
                    cv2.setTrackbarPos(param.name, self.proc_window_name,
                                       param.multiplier)
                    self.filters[self.filter_type].validated = False
                break
        self.filters[self.filter_type].kernel = self.filters[self.filter_type].update()
        if self.write_tracking_data:
            self.write_tracking_data = False
            self.writeResults()
        self.reset = True

    def getParamStrings(self):
        dataset_params = ''
        if self.from_cam:
            dataset_params = 'cam'
        else:
            start_id = self.labels.index('type')
            for i in xrange(start_id, len(self.init_params)):
                dataset_params = dataset_params + '_' + self.init_params[i]
            dataset_params = dataset_params + '_%d' % (self.times + 1)
        filter_id = 'none'
        filter_param_str = ''
        if self.filter_type != 'none':
            filter_id = self.filters[self.filter_type].type
            for key in self.filters[self.filter_type].params.keys():
                filter_param_str = filter_param_str + '_' + str(self.filters[self.filter_type].params[key].val)
        filter_param_str = filter_param_str.replace('.', 'd')

        tracker_param_str = ''
        # tracker_id=self.trackers[self.tracker_type].type
        # print 'tracker_id=', tracker_id
        try:
            params = self.trackers[self.tracker_type].params
            for i in xrange(len(params['trackers'].val)):
                tracker_type = params['trackers'].val[i]
                if tracker_type == 'none':
                    continue
                tracker_param_str = tracker_param_str + '-' + tracker_type
                # tracker_params = params['parameters'].val[i].params
                # for key in tracker_params.keys():
                # param_val=tracker_params[key].val
                # tracker_param_str = tracker_param_str +'_' + str(param_val)
        except KeyError:
            for key in self.trackers[self.tracker_type].params.keys():
                tracker_param_str = tracker_param_str + '_' + str(self.trackers[self.tracker_type].params[key].val)

        tracker_param_str = tracker_param_str.replace('.', 'd')

        return [dataset_params, filter_id, filter_param_str, tracker_param_str]

    def writeResults(self):
        return
        if self.times <= 1:
            return
        print('Saving results...')
        [dataset_params, filter_id, filter_params, tracking_params] = self.getParamStrings()
        self.max_fps = max(self.curr_fps_list[1:])
        min_fps = min(self.curr_fps_list[1:])
        self.max_error = max(self.curr_error_list)

        if self.batch_mode:
            tracking_res_dir = 'Results/batch'
        else:
            tracking_res_dir = 'Results'

        if not os.path.isdir(tracking_res_dir):
            os.makedirs(tracking_res_dir)

        tracking_res_fname = tracking_res_dir + '/summary.txt'

        if not os.path.exists(tracking_res_fname):
            res_file = open(tracking_res_fname, 'a')
            res_file.write(
                "tracker".ljust(10) +
                "\tcolor_space".ljust(10) +
                "\tfilter".ljust(10) +
                "\tmultichannel".ljust(15) +
                "\tSCV".ljust(10) +
                "\tavg_error".rjust(14) +
                "\tmax_error".rjust(14) +
                "\tsuccess".rjust(14) +
                "\tdrift".rjust(14) +
                "\tavg_fps".rjust(14) +
                "\tmax_fps".rjust(14) +
                "\tmin_fps".rjust(14) +
                "\tdataset".center(50) +
                "\ttracking params".center(100) +
                "\tfilter params".center(50) + '\n'
            )
        else:
            res_file = open(tracking_res_fname, 'a')

        success_rate = float(self.success_count) / float(self.times + 1) * 100
        if self.success_count > 0:
            drift = sum(self.success_drift) / float(self.success_count)
        else:
            drift = -1
        try:
            multi_approach = self.tracker.multi_approach
            use_scv = self.tracker.use_scv
        except AttributeError:
            sub_tracker1 = self.tracker.trackers[0]
            multi_approach = sub_tracker1.multi_approach
            use_scv = sub_tracker1.use_scv

        print 'multi_approach=', multi_approach
        print 'filter_id=', filter_id
        print 'color_space=', self.color_space
        print 'tracker_type=', self.tracker_type

        res_file.write(
            self.tracker_type.ljust(10) +
            "\t" + self.color_space.ljust(10) +
            "\t" + filter_id.ljust(10) +
            "\t" + multi_approach.ljust(15) +
            "\t" + str(use_scv).ljust(10) +
            "\t%13.6f" % self.avg_error +
            "\t%13.6f" % self.max_error +
            "\t%13.6f" % success_rate +
            "\t%13.6f" % drift +
            "\t%13.6f" % self.average_fps +
            "\t%13.6f" % self.max_fps +
            "\t%13.6f" % min_fps +
            "\t" + dataset_params.center(50) +
            "\t" + tracking_params.center(100) +
            "\t" + filter_params.center(50) + "\n"
        )
        res_file.close()

        print 'success rate:', success_rate
        print 'average error:', self.avg_error
        print 'average fps:', self.average_fps
        print 'average drift:', drift

        if self.avg_filename is not None and self.agg_filename is not None:
            print 'writing avg data to ', 'Results/' + self.avg_filename + '.txt'
            avg_full_name = 'Results/' + self.avg_filename + '.txt'
            if not os.path.exists(avg_full_name):
                avg_file = open(avg_full_name, 'a')
                avg_file.write(
                    "parameters".center(len(self.agg_filename)) +
                    "\tsuccess_rate".center(14) +
                    "\tavg_fps".center(14) +
                    "\tavg_drift\n".center(14)
                )
            else:
                avg_file = open(avg_full_name, 'a')
            avg_file.write(
                self.agg_filename +
                "\t%13.6f" % success_rate +
                "\t%13.6f" % self.average_fps +
                "\t%13.6f\n" % drift
            )
            avg_file.close()
        self.savePlots(dataset_params, filter_id, filter_params, tracking_params)
        # self.tracking_data_file.close()
        # webbrowser.open(tracking_res_fname)

    def generateCombinedPlots(self):
        combined_fig = plt.figure(1)
        plt.subplot(211)
        plt.title('Tracking Error')
        plt.ylabel('Error')
        plt.plot(self.frame_times, self.avg_error_list, 'r',
                 self.frame_times, self.curr_error_list, 'g')

        plt.subplot(212)
        plt.title('FPS')
        plt.xlabel('Frame')
        plt.ylabel('FPS')
        plt.plot(self.frame_times, self.avg_fps_list, 'r',
                 self.frame_times, self.curr_fps_list, 'g')
        return combined_fig

    def savePlots(self, dataset_params, filter_id, filter_params, tracking_params):
        print('Saving plot data...')
        if self.batch_mode:
            res_dir = 'Results/batch/' + self.tracker_type + '/' + filter_id
        else:
            res_dir = 'Results/' + self.tracker_type + '/' + filter_id
        plot_dir = res_dir + '/plots'

        res_template = dataset_params + '_' + filter_params + '_' + self.color_space + '_' + \
                       self.smoothing_type + '_' + str(
            self.smoothing_kernel) + '_' + tracking_params + '_' + self.feature
        print 'res_template=', res_template
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        plot_fname = plot_dir + '/' + res_template
        combined_fig = self.generateCombinedPlots()
        combined_fig.savefig(plot_fname, ext='png', bbox_inches='tight')
        plt.figure(0)

        res_fname = res_dir + '/' + res_template + '.txt'
        res_file = open(res_fname, 'w')
        res_file.write(tracking_params + '\n')
        res_file.write("curr_fps".rjust(10) + "\t" + "avg_fps".rjust(10) + "\t\t" +
                       "curr_error".rjust(10) + "\t" + "avg_error".rjust(10) + "\n")
        for i in xrange(len(self.avg_fps_list)):
            res_file.write("%10.5f\t" % self.curr_fps_list[i] +
                           "%10.5f\t\t" % self.avg_fps_list[i] +
                           "%10.5f\t" % self.curr_error_list[i] +
                           "%10.5f\n" % self.avg_error_list[i])
        res_file.close()
        getThresholdVariations(res_dir, res_template, 'error', show_plot=False,
                               min_thresh=0, diff=1, max_thresh=100, max_rate=100,
                               agg_filename=self.agg_filename)
        # getThresholdVariations(res_dir, res_template, 'fps', show_plot=False,
        # min_thresh=0, diff=1, max_thresh=30, max_rate=100,
        # agg_filename=self.agg_filename)

    def cleanup(self):
        self.tracking_data_file.close()

        # def applyFiltering(self):
        # if self.filter_type == 'none':
        # proc_img = self.gray_img
        # elif self.filter_type == 'DoG' or \
        # self.filter_type == 'gauss' or \
        # self.filter_type == 'bilateral' or \
        # self.filter_type == 'median' or \
        # self.filter_type == 'canny':
        # proc_img = self.filters[self.filter_type].apply(self.gray_img)
        # elif self.filter_type in self.filters.keys():
        # proc_img = self.filters[self.filter_type].apply(self.gray_img_float)
        # else:
        # print "Invalid filter type ", self.filter_type
        # return None
        # return proc_img

