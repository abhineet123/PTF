# import threading
# from multiprocessing import Process
# from pathos.multiprocessing import ProcessingPool
# import matplotlib
# matplotlib.use('QT4Agg')
from matplotlib import pyplot as plt
from matplotlib import animation
from InteractiveTracking_mt import InteractiveTrackingApp
import init_mt as init
from TrackingParams import TrackingParams
import sys
import os
import cv2
from Misc import *
from ImageUtils import draw_region
import time
import xvInput


class StandaloneTrackingApp(InteractiveTrackingApp):
    """ A demo program that uses OpenCV to grab frames. """

    def __init__(self, init_frame, root_path,
                 params, tracking_params, filtering_params, labels, default_id,
                 buffer_size, success_threshold=5, batch_mode=False,
                 agg_filename=None, avg_filename=None, anim_app=None):
        track_window_name = 'Tracked Images'
        InteractiveTrackingApp.__init__(self, init_frame, root_path, track_window_name, params,
                                        tracking_params, filtering_params, labels, default_id,
                                        success_threshold, batch_mode, agg_filename, avg_filename,
                                        anim_app)
        self.buffer_id = None
        self.buffer_end_id = -1
        self.buffer_start_id = 0
        self.buffer_full = False
        self.buffer_size = buffer_size
        self.current_buffer_size = 0
        self.raw_frame_buffer = []
        self.proc_frame_buffer = []
        self.buffer_accesses = [0] * self.buffer_size
        self.current_corners_buffer = []
        self.actual_corners_buffer = []
        self.rewind = False
        self.switch_plot = False
        self.from_frame_buffer = False
        self.plot_index = 0

    def run(self):
        cv2.namedWindow(self.track_window_name)
        # cv2.namedWindow(self.proc_window_name)

        self.start_time = time.clock()
        self.current_time = time.clock()

        frame_id = self.init_frame_id
        while True:
            if not self.keyboardHandler():
                sys.exit()
            # print 'frame_id: ', frame_id
            self.src_img = self.getRawFrame()
            if self.src_img is None:
                sys.exit()
            self.annotated_img = self.src_img.copy()
            self.proc_img = self.getProcessedFrame(self.src_img)
            cv2.imshow(self.proc_window_name, self.proc_img)

            self.proc_img = self.proc_img.astype(np.float64)

            for tracker_id in xrange(self.no_of_trackers):
                if not self.updateTracker(tracker_id):
                    print 'Done tracking'
                    sys.exit()

            self.last_time = self.current_time
            self.current_time = time.clock()

            self.average_fps = (frame_id - self.init_frame_id + 1) / (self.current_time - self.start_time)
            self.current_fps = 1.0 / (self.current_time - self.last_time)

            fps_text = "%5.2f" % self.average_fps + "   %5.2f" % self.current_fps
            cv2.putText(self.annotated_img, fps_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

            cv2.imshow(self.track_window_name, self.annotated_img)
            frame_id += 1

    def initFrameBuffer(self):
        print 'Initializing frame buffer with ', self.buffer_size, ' frames...'
        for frame_id in xrange(self.buffer_size):
            raw_frame = self.getRawFrame()
            if raw_frame is None:
                break
            proc_frame = self.getProcessedFrame(raw_frame)
            self.raw_frame_buffer.append(raw_frame.copy())
            self.proc_frame_buffer.append(proc_frame.copy())
        print 'Done'


    def runParallel(self):

        # self.initFrameBuffer()
        self.buffer_id = [-1] * self.no_of_trackers
        cv2.namedWindow(self.track_window_name)
        self.start_time = time.clock()
        self.current_time = time.clock()

        pool = ProcessingPool(nodes=2)
        result = pool.map(lambda x: self.runTracker(x), range(self.no_of_trackers))

        # tracker_procs = []
        # for tracker_id in xrange(self.no_of_trackers):
        # # self.current_corners_buffer.append([])
        #     proc = mp.Process(target=self.runTracker, args=(tracker_id,))
        #     tracker_procs.append(proc)
        # for proc in tracker_procs:
        #     proc.start()
        #     proc.join()


    def runTracker(self, tracker_id):
        tracker = self.trackers[tracker_id]
        tracker_col = self.tracker_cols[tracker_id]
        tracker_type = self.tracker_types[tracker_id]

        while True:
            if not self.keyboardHandler():
                sys.exit()
            self.buffer_id[tracker_id] += 1
            buffer_id = self.buffer_id[tracker_id]
            if buffer_id < self.current_buffer_size:
                proc_img = self.proc_frame_buffer[buffer_id]
                annotated_img = self.raw_frame_buffer[buffer_id]
            else:
                annotated_img = self.getRawFrame(buffer_id)
                proc_img = self.getProcessedFrame(annotated_img)
                self.raw_frame_buffer.append(annotated_img)
                self.proc_frame_buffer.append(proc_img)

            self.buffer_accesses[buffer_id] += 1
            if not tracker.initialized:
                init_location = self.init_locations[tracker_id]
                init_array = np.array(init_location, dtype=np.float64).T
                tracker.initialize(self.proc_img, init_array)

            tracker.update(proc_img)
            current_corners = tracker.get_region()
            center_pt = np.mean(current_corners, axis=1)
            center_pt = tuple([int(x) for x in center_pt])

            cv2.putText(annotated_img, tracker_type,
                        center_pt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, tracker_col)
            draw_region(annotated_img, current_corners, tracker_col, 2)

            if self.buffer_accesses[buffer_id] < self.no_of_trackers:
                continue

            self.last_time = self.current_time
            self.current_time = time.clock()

            self.average_fps = (buffer_id - self.init_frame_id + 1) / (self.current_time - self.start_time)
            self.current_fps = 1.0 / (self.current_time - self.last_time)

            fps_text = "%5.2f" % self.average_fps + "   %5.2f" % self.current_fps
            cv2.putText(self.annotated_img, fps_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

            cv2.imshow(self.track_window_name, self.annotated_img)


    def updateTracker(self, tracker_id):

        tracker = self.trackers[tracker_id]

        if not tracker.is_initialized():
            init_location = self.init_locations[tracker_id]
            init_array = np.array(init_location, dtype=np.float64).T
            tracker.initialize(self.proc_img, init_array)

        tracker_col = self.tracker_cols[tracker_id]
        tracker.update(self.proc_img)
        self.current_corners = tracker.get_region()
        center_pt = np.mean(self.current_corners, axis=1)
        center_pt = [int(x) for x in center_pt]
        # print 'center_pt=', center_pt
        tracker_text = self.tracker_types[tracker_id]
        if tracker_text == 'xv_ssd':
            tracker_text = tracker.text
        cv2.putText(self.annotated_img, tracker_text,
                    tuple(center_pt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, tracker_col)
        draw_region(self.annotated_img, self.current_corners, tracker_col, 2)


        # if not self.on_frame(proc_img, i):
        # return False
        # self.frame_times.append(i)
        # self.curr_error_list.append(self.curr_error)
        # self.avg_error = np.mean(self.curr_error_list)
        # self.avg_error_list.append(self.avg_error)
        # self.curr_fps_list.append(self.current_fps)
        # self.avg_fps_list.append(self.average_fps)
        return True


    def onKeyPress(self, event):
        print 'key pressed=', event.key
        if event.key == "escape":
            self.exit_event = True
            sys.exit()
        elif event.key == "shift":
            if not self.from_cam or True:
                self.switch_plot = True
                self.plot_fps = not self.plot_fps
        elif event.key == " ":
            self.paused = not self.paused


    def keyboardHandler(self):
        key = cv2.waitKey(1)
        if key != -1:
            print "key=", key
        if key == ord(' '):
            self.paused = not self.paused
        elif key == 27:
            return False
        elif key == ord('p') or key == ord('P'):
            if not self.from_cam or True:
                self.switch_plot = True
                self.plot_fps = not self.plot_fps
        elif key == ord('w') or key == ord('W'):
            self.write_res = not self.write_res
            if self.write_res:
                print "Writing results enabled"
            else:
                print "Writing results disabled"
        elif key == ord('t') or key == ord('T'):
            self.tracker_pause = not self.tracker_pause
        elif key == ord('r') or key == ord('R'):
            if self.from_frame_buffer:
                self.rewind = not self.rewind
                if self.rewind:
                    print "Disabling rewind"
                    # self.from_frame_buffer=False
                    # self.rewind=False
                else:
                    print "Enabling rewind"
                    # self.rewind=True
            else:
                print "Switching to frame buffer"
                print "Enabling rewind"
                self.from_frame_buffer = True
                self.rewind = True
                self.buffer_id = self.buffer_end_id
        return True


    def updatePlots(self, frame_count):
        if self.from_cam:
            ax.set_xlim(0, frame_count)
        if self.switch_plot:
            self.switch_plot = False
            # print "here we are"
            if self.plot_fps:
                fig.canvas.set_window_title('FPS')
                plt.ylabel('FPS')
                plt.title('FPS')
                self.max_fps = max(self.curr_fps_list)
                ax.set_ylim(0, self.max_fps)
            else:
                fig.canvas.set_window_title('Tracking Error')
                plt.ylabel('Error')
                plt.title('Tracking Error')
                self.max_error = max(self.curr_error_list)
                ax.set_ylim(0, self.max_error)
            plt.draw()

        if self.plot_fps:
            line1.set_data(self.frame_times[0:self.plot_index + 1], self.avg_fps_list[0:self.plot_index + 1])
            line2.set_data(self.frame_times[0:self.plot_index + 1], self.curr_fps_list[0:self.plot_index + 1])
            # line3.set_data(self.frame_times[self.plot_index],self.curr_fps_list[self.plot_index])
            if max(self.curr_fps_list) > self.max_fps:
                self.max_fps = max(self.curr_fps_list)
                ax.set_ylim(0, self.max_fps)
                plt.draw()
        else:
            # line1.set_data(self.frame_times[0:self.plot_index + 1], self.avg_error_list[0:self.plot_index + 1])
            line1.set_data(self.frame_times[0:self.plot_index + 1], self.update_diff[0:self.plot_index + 1])
            line2.set_data(self.frame_times[0:self.plot_index + 1], self.curr_error_list[0:self.plot_index + 1])

            if max(self.curr_error_list) > self.max_error:
                self.max_error = max(self.curr_error_list)
                max_update_diff = max(self.update_diff)
                ax.set_ylim(0, max(self.max_error, max_update_diff))
                plt.draw()


    def updatePlots2(self):
        pass
        hist = self.tracker.feature_obj.hist
        if hist is not None:
            plt.plot(np.arange(hist.shape[0]), hist)


    def animate(self, i):
        if not self.keyboardHandler():
            self.writeResults()
            sys.exit()

        if self.paused:
            return line1, line2

        if not self.buffer_full:
            if len(self.frame_buffer) >= self.buffer_size:
                print "Frame buffer full"
                # print "buffer_end_id=", self.buffer_end_id
                # print "buffer_start_id=", self.buffer_start_id
                self.buffer_full = True

        if self.from_frame_buffer:
            if self.rewind:
                self.buffer_id -= 1
                self.plot_index -= 1
                if self.buffer_id < 0:
                    self.buffer_id = self.buffer_size - 1
                elif self.buffer_id == self.buffer_start_id:
                    print "Disabling rewind"
                    self.rewind = False
            else:
                self.buffer_id += 1
                self.plot_index += 1
                if self.buffer_id >= self.buffer_size:
                    self.buffer_id = 0
                elif self.buffer_id == self.buffer_end_id:
                    self.from_frame_buffer = False
                    print "Getting back to video stream"
            self.src_img = self.frame_buffer[self.buffer_id]
            self.current_corners = self.current_corners_buffer[self.buffer_id]
            self.actual_corners = self.actual_corners_buffer[self.buffer_id]
        else:
            self.plot_index = i
            if not self.updateTracker(i):
                self.writeResults()
                sys.exit()
            if not self.buffer_full:
                self.frame_buffer.append(self.src_img.copy())
                self.current_corners_buffer.append(self.current_corners.copy())
                self.actual_corners_buffer.append(self.actual_corners.copy())
                self.buffer_end_id += 1
            else:
                self.frame_buffer[self.buffer_start_id] = self.src_img.copy()
                self.current_corners_buffer[self.buffer_start_id] = self.current_corners.copy()
                self.actual_corners_buffer[self.buffer_start_id] = self.actual_corners.copy()
                self.buffer_end_id = self.buffer_start_id
                self.buffer_start_id = (self.buffer_start_id + 1) % self.buffer_size

        if self.src_img is not None:
            self.display()
        if use_plot:
            self.updatePlots(i)
        # self.updatePlots2()
        return line1, line2


def simData():
    i = -1
    while not app.exit_event:
        if not app.paused and not app.from_frame_buffer:
            i += 1
        if app.reset:
            print "Resetting the plots..."
            ax.cla()
            plt.draw()
            i = 0
        print 'simData: i=', i
        yield i


def processArguments(args):
    no_of_args = len(args)
    if no_of_args % 2 != 0:
        print 'args=\n', args
        raise SystemExit('Error in processArguments: '
                         'Optional arguments need to be specified in pairs')

    compound_trackers = ['cascade']
    compound_filters = ['cascade']
    tracker_index = labels.index('tracker')
    filter_index = labels.index('filter')
    agg_filename = None
    avg_filename = None

    for i in xrange(no_of_args / 2):
        arg = sys.argv[i * 2 + 1].split('::')
        print 'arg=', arg
        arg_type = arg[0]
        arg_label = arg[1]
        arg_val = sys.argv[i * 2 + 2]
        # print 'arg_label=', arg_label
        # print 'arg_val=', arg_val
        if arg_type == 'basic':
            arg_index = labels.index(arg_label)
            # print 'arg_index=', arg_index
            if arg_label == 'task':
                task_types = basic_params[labels.index('type')]
                simple_tasks = basic_params[labels.index('task')][task_types.index('simple')]
                complex_tasks = basic_params[labels.index('task')][task_types.index('complex')]
                if arg_val in simple_tasks:
                    task_id = 0
                elif arg_val in complex_tasks:
                    task_id = 1
                else:
                    raise SystemExit('Error in processArguments:'
                                     'Invalid task provided: ' + arg_val)
                default_id[labels.index('type')] = task_id
                default_id[arg_index] = basic_params[arg_index][task_id].index(arg_val)
            else:
                default_id[arg_index] = basic_params[arg_index].index(arg_val)
        elif arg_type == 'tracker':
            current_tracker = basic_params[tracker_index][default_id[tracker_index]]
            curr_tracking_params = tracking_params[current_tracker]
            if arg_label not in curr_tracking_params.keys():
                raise SystemExit('Error in processArguments:'
                                 'Invalid argument ' + arg_label + ' provided')

            param_type = curr_tracking_params[arg_label]['type']
            if param_type == 'int':
                arg_val = int(arg_val)
            elif param_type == 'float':
                arg_val = float(arg_val)
            elif param_type == 'boolean':
                if arg_val.lower() == 'true':
                    arg_val = True
                elif arg_val.lower() == 'false':
                    arg_val = False
                else:
                    raise SystemExit('Error in processArguments:'
                                     'Invalid value ' + arg_val +
                                     ' specified for parameter ' + arg_label)
            curr_tracking_params[arg_label]['default'] = arg_val
        elif arg_type == 'filter':
            current_filter = basic_params[filter_index][default_id[filter_index]]
            if current_filter == 'none':
                curr_filtering_params = {}
            else:
                curr_filtering_params = filtering_params[current_filter]
            if arg_label not in curr_filtering_params.keys():
                raise SystemExit('Error in processArguments:'
                                 'Invalid argument ' + arg_label + ' provided')
            # param_type=curr_filtering_params[arg_label]['type']
            # if param_type=='int':
            # arg_val=int(arg_val)
            # elif param_type=='float':
            # arg_val=float(arg_val)
            arg_val = int(arg_val)
            # param_base=current_filter[arg_label]['default']['base']
            # param_add=current_filter[arg_label]['type']['add']
            param_limit = curr_filtering_params[arg_label]['default']['limit']
            # mult=(arg_val-param_add)/param_base
            mult = arg_val
            if mult > param_limit:
                raise SystemExit(('Error in processArguments:'
                                  'Specified value ' + str(arg_val) + 'for parameter ' + arg_label +
                                  'exceeds the maximum allowed value ' + str(param_limit)))
            curr_filtering_params[arg_label]['default']['mult'] = mult
        elif arg_type == 'compound_tracker':
            current_tracker = basic_params[tracker_index][default_id[tracker_index]]
            if current_tracker not in compound_trackers:
                raise SystemExit('Error in processArguments:'
                                 'Compound tracker variable ' + arg_label +
                                 ' specified for simple tracker ' + current_tracker)
            curr_tracking_params = tracking_params[current_tracker]
            if len(arg) < 3:
                raise SystemExit('Error in processArguments:'
                                 'ID not specified for compound tracker variable ' + arg_label)
            arg_id = int(arg[2])
            if len(arg) < 4:
                curr_tracking_params[arg_label]['default'][arg_id] = arg_val
            else:
                param_label = arg[3]
                current_params = curr_tracking_params[arg_label]['default'][arg_id]
                if current_params is None:
                    sub_tracker = curr_tracking_params['tracker']['default'][arg_id]
                    sub_params = tracking_params[sub_tracker].copy()
                    curr_tracking_params[arg_label]['default'][arg_id] = TrackingParams(sub_tracker, sub_params)

                curr_tracking_params[arg_label]['default'][arg_id][param_label]['default'] = arg_val

        elif arg_type == 'misc':
            if arg_label == 'aggregate':
                agg_filename = arg_val
            elif arg_label == 'average':
                avg_filename = arg_val
            else:
                raise SystemExit('Error in processArguments:'
                                 'Invalid argument ' + arg_label + ' provided')

    return agg_filename, avg_filename


def AnimatedPlot():
    print "Starting AnimatedPlot"
    fig = plt.figure(0)
    fig.canvas.set_window_title('Tracking Error')
    cid = fig.canvas.mpl_connect('key_press_event', app.onKeyPress)
    ax = plt.axes(xlim=(0, app.no_of_frames), ylim=(0, 5))
    plt.xlabel('Frame')
    # plt.ylabel('Error')
    # plt.title('Tracking Error')
    plt.grid(True)
    line1, line2 = ax.plot([], [], 'r', [], [], 'g')
    plt.legend(('Average', 'Current'))
    # plt.draw()
    def initPlot():
        line1.set_data([], [])
        line2.set_data([], [])
        # line3.set_data(0, 0)
        return line1, line2

    anim = animation.FuncAnimation(fig, app.animate, simData, init_func=initPlot,
                                   interval=0, blit=True)
    plt.show()
    print "Exiting AnimatedPlot "


if __name__ == '__main__':
    init_frame = 0
    success_threshold = 5
    frame_buffer_size = 1000
    db_root_path = '/home/abhineet/G/UofA/Thesis/#Code/Datasets'

    [basic_params, labels, default_id] = init.getBasicParams()
    tracking_params = init.getTrackingParams()
    filtering_params = init.getFilteringParams()

    app = StandaloneTrackingApp(init_frame, db_root_path, basic_params, tracking_params,
                                filtering_params, labels, default_id, frame_buffer_size,
                                success_threshold)

    app.run()
    # app.runParallel()
    # anim_app=animatedPlot()





