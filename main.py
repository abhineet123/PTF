# import matplotlib
# matplotlib.use('QT4Agg')
from matplotlib import pyplot as plt
from matplotlib import animation
from InteractiveTracking import InteractiveTrackingApp
import config
from FilteringParams import *
from TrackingParams import *
import sys

xv_input_found = True
try:
    import CModules.xvInput as xvInput
except:
    xv_input_found = False
import time


class StandaloneTrackingApp(InteractiveTrackingApp):
    """ A demo program that uses OpenCV to grab frames. """

    def __init__(self, init_frame, root_path,
                 params, tracking_params, filtering_params, labels, default_id,
                 buffer_size, success_threshold=5, batch_mode=False,
                 agg_filename=None, avg_filename=None, anim_app=None, extended_db=False,
                 write_tracking_data=False, tracking_data_fname=None, camera_id = 0):
        track_window_name = 'Tracked Images - Press space to pause and r to rewind'
        InteractiveTrackingApp.__init__(self, init_frame, root_path, track_window_name, params,
                                        tracking_params, filtering_params, labels, default_id,
                                        success_threshold, batch_mode, agg_filename, avg_filename,
                                        anim_app, extended_db, write_tracking_data, tracking_data_fname, camera_id)
        self.buffer_id = 0
        self.buffer_end_id = -1
        self.buffer_start_id = 0
        self.buffer_full = False
        self.buffer_size = buffer_size
        self.current_buffer_size = 0
        self.frame_buffer = []
        self.current_corners_buffer = []
        self.actual_corners_buffer = []
        self.rewind = False
        self.switch_plot = False
        self.from_frame_buffer = False
        self.plot_index = 0
        self.cam_skip_frames = 50


    def exit(self):
        print "avg_fps: ", self.average_fps
        print "avg_fps_with_input: ", self.average_fps_win
        self.exit_event = True
        sys.exit()

    def run(self, show_img=False, write_tracking_data=True):
        global pause_after_frame
        if self.batch_mode and self.source == 'camera':
            raise SystemExit('Batch mode cannot be run with camera')
        frame_id = self.init_frame
        while True:
            # if not self.keyboardHandler():
            # self.writeResults()
            # self.exit()
            if not self.updateTracker(frame_id):
                # self.writeResults()
                self.exit()
            if show_img:
                if self.img is not None:
                    self.display()
                key = cv2.waitKey(1 - pause_after_frame)
                if key == ord(' '):
                    pause_after_frame = 1 - pause_after_frame
                elif key == 27:
                    self.exit()
            frame_id += 1

    def updateTracker(self, frame_id):
        if self.reset:
            self.reset = False
            self.inited = False
            self.initPlotParams()

        if self.exit_event:
            # self.tracker.cleanup()
            self.exit_event = True
            return False

        if not self.from_cam and self.init_method == 'ground_truth' and frame_id >= self.no_of_frames:
            print 'Reached end of video stream'
            self.exit_event = True
            return False

        self.start_time_win = time.clock()
        if xv_input_found and self.pipeline == 'XVision':
            # xvInput.getFrame(self.src_img)
            self.src_img = xvInput.getFrame2(0)
        else:
            ret, self.src_img = self.cap.read()
            if not ret:
                print "Frame could not be read from OpenCV pipeline"
                return False
            if self.from_cam and not self.inited:
                print "Skipping ", self.cam_skip_frames, " frames...."
                for j in xrange(self.cam_skip_frames):
                    ret, self.src_img = self.cap.read()

        if not self.on_frame(self.src_img, frame_id):
            return False

        self.current_fps = 1.0 / (self.end_time - self.start_time)
        self.current_fps_win = 1.0 / (self.end_time - self.start_time_win)
        # self.average_fps = np.mean(np.asarray(self.curr_fps_list))
        # self.average_fps_win = np.mean(np.asarray(self.curr_fps_win_list))
        if frame_id > self.init_frame:
            self.average_fps += (self.current_fps - self.average_fps) / frame_id
            self.average_fps_win += (self.current_fps_win - self.average_fps_win) / frame_id
            self.avg_error += (self.curr_error - self.avg_error) / frame_id

        self.frame_times.append(frame_id)

        self.curr_fps_list.append(self.current_fps)
        self.avg_fps_list.append(self.average_fps)
        # self.curr_fps_win_list.append(self.current_fps_win)

        # self.avg_error = np.mean(self.curr_error_list)
        self.curr_error_list.append(self.curr_error)
        self.avg_error_list.append(self.avg_error)

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
            # elif event.key == "ctrl+r":
            # self.paused=True
            # self.gui_obj.initWidgets('Restart')

    def keyboardHandler(self):
        global pause_after_frame
        key = cv2.waitKey(1)
        if key != -1:
            print "key=", key
        if key == ord(' '):
            pause_after_frame = 1 - pause_after_frame
            # self.paused = not self.paused
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
        elif key == ord('m') or key == ord('M'):
            self.gui_obj.initWidgets(start_label='Restart')

        return True


    def updatePlots(self, frame_count):
        if self.from_cam:
            ax.set_xlim(0, frame_count)
        if self.switch_plot:
            self.switch_plot = False
            # print "here we are"
            if self.plot_fps:
                fig.canvas.set_window_title('Press shift to switch to tracking error')
                plt.ylabel('FPS')
                plt.title('FPS')
                self.max_fps = max(self.curr_fps_list)
                ax.set_ylim(0, self.max_fps)
            else:
                fig.canvas.set_window_title('Press shift to switch to FPS')
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
            line1.set_data(self.frame_times[0:self.plot_index + 1], self.avg_error_list[0:self.plot_index + 1])
            # line1.set_data(self.frame_times[0:self.plot_index + 1], self.update_diff[0:self.plot_index + 1])
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
            # self.writeResults()
            self.exit()

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
            self.img = self.frame_buffer[self.buffer_id]
            self.current_corners = self.current_corners_buffer[self.buffer_id]
            self.actual_corners = self.actual_corners_buffer[self.buffer_id]
        else:
            self.plot_index = i
            if not self.updateTracker(i):
                # self.writeResults()
                self.exit()
            if not self.buffer_full:
                self.frame_buffer.append(self.img.copy())
                self.current_corners_buffer.append(self.current_corners.copy())
                self.actual_corners_buffer.append(self.actual_corners.copy())
                self.buffer_end_id += 1
            else:
                self.frame_buffer[self.buffer_start_id] = self.img.copy()
                self.current_corners_buffer[self.buffer_start_id] = self.current_corners.copy()
                self.actual_corners_buffer[self.buffer_start_id] = self.actual_corners.copy()
                self.buffer_end_id = self.buffer_start_id
                self.buffer_start_id = (self.buffer_start_id + 1) % self.buffer_size

        if self.img is not None:
            self.display()
            # cv2.waitKey(1-pause_after_frame)
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
        yield i


def processArguments(args):
    global pause_after_frame, init_frame, db_root_path, show_img, \
        write_tracking_data, tracking_data_fname, extended_db, agg_filename, avg_filename

    no_of_args = len(args)

    if no_of_args == 1:
        return
    elif no_of_args % 2 != 0:
        print 'args=\n', args
        print 'processArguments: ' \
              'Command line arguments need to be specified in pairs'
        return

    compound_trackers = ['cascade']
    compound_filters = ['cascade']
    tracker_index = labels.index('tracker')
    filter_index = labels.index('filter')

    for arg_id in xrange(no_of_args / 2):
        arg = sys.argv[arg_id * 2 + 1].split('::')
        arg_type = arg[0]
        arg_label = arg[1]
        arg_val = sys.argv[arg_id * 2 + 2]
        print 'arg_type: ', arg_type, '\targ_label: ', arg_label, '\targ_val: ', arg_val
        if arg_type == 'ptf':
            if arg_label == 'pause_after_frame':
                pause_after_frame = int(arg_val)
            elif arg_label == 'init_frame':
                init_frame = int(arg_val)
            elif arg_label == 'db_root_path':
                db_root_path = arg_val
            elif arg_label == 'show_img':
                show_img = int(arg_val)
            elif arg_label == 'write_tracking_data':
                write_tracking_data = int(arg_val)
            elif arg_label == 'tracking_data_fname':
                tracking_data_fname = arg_val
            elif arg_label == 'extended_db':
                extended_db = int(arg_val)
            elif arg_label == 'aggregate':
                agg_filename = arg_val
            elif arg_label == 'average':
                avg_filename = arg_val
            else:
                print 'processArguments: Invalid argument ' + arg_label + ' provided'
                continue
        elif arg_type == 'basic':
            arg_index = labels.index(arg_label)
            # print 'arg_index=', arg_index
            if arg_label == 'task':
                task_types = basic_params[labels.index('type')]
                valid_task = False
                for actor_id in xrange(len(task_types)):
                    if arg_val in basic_params[labels.index('task')][actor_id]:
                        task_id = actor_id
                        valid_task = True
                        break
                if not valid_task:
                    print 'processArguments: Invalid task provided: ' + arg_val
                    continue
                default_id[labels.index('type')] = task_id
                default_id[arg_index] = basic_params[arg_index][task_id].index(arg_val)
            else:
                default_id[arg_index] = basic_params[arg_index].index(arg_val)
        elif arg_type == 'id':
            # specify the default id directly
            arg_index = labels.index(arg_label)
            default_id[arg_index] = int(arg_val)
        elif arg_type == 'tracker':
            current_tracker = basic_params[tracker_index][default_id[tracker_index]]
            curr_tracking_params = tracking_params[current_tracker]
            if arg_label not in curr_tracking_params.keys():
                print 'processArguments: Invalid argument ' + arg_label + ' provided'
                continue
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
                    print 'processArguments: Invalid value ' + arg_val + \
                          ' specified for parameter ' + arg_label
                    continue
            curr_tracking_params[arg_label]['default'] = arg_val
        elif arg_type == 'filter':
            current_filter = basic_params[filter_index][default_id[filter_index]]
            if current_filter == 'none':
                curr_filtering_params = {}
            else:
                curr_filtering_params = filtering_params[current_filter]
            if arg_label not in curr_filtering_params.keys():
                print 'processArguments: Invalid argument ' + arg_label + ' provided'
                continue
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
                print 'processArguments: Specified value ' \
                      + str(arg_val) + 'for parameter ' + arg_label \
                      + ' exceeds the maximum allowed value ' + str(param_limit)
                continue
            curr_filtering_params[arg_label]['default']['mult'] = mult
        elif arg_type == 'compound_tracker':
            current_tracker = basic_params[tracker_index][default_id[tracker_index]]
            if current_tracker not in compound_trackers:
                print 'processArguments: Compound tracker variable ' + arg_label \
                      + ' specified for simple tracker ' + current_tracker
                continue
            curr_tracking_params = tracking_params[current_tracker]
            if len(arg) < 3:
                print 'processArguments:ID not specified for compound tracker variable ' + arg_label
                continue
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

    return agg_filename, avg_filename


class animatedPlot:
    def __init__(self):
        self.start_anim = False

    def startAnimation(self):
        while not self.start_anim:
            # print 'Not starting yet'
            cv2.waitKey(1)
            continue
        print 'Starting now !'
        fig = plt.figure(0)
        fig.canvas.set_window_title('Press shift to switch to FPS')
        cid = fig.canvas.mpl_connect('key_press_event', app.onKeyPress)
        ax = plt.axes(xlim=(0, app.no_of_frames), ylim=(0, 5))
        plt.xlabel('Frame')
        plt.ylabel('Error')
        plt.title('Tracking Error')
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


if __name__ == '__main__':

    pause_after_frame = 0
    init_frame = 0
    success_threshold = 5
    frame_buffer_size = 1000
    db_root_path = '../Datasets'
    camera_id = 1
    agg_filename = None
    avg_filename = None
    batch_mode = False
    show_img = False
    write_tracking_data = True
    tracking_data_fname = None
    extended_db = 1

    tracking_params = config.getTrackingParams(flann_found, cython_found,
                                               xvision_found, mtf_found)
    trackers = sorted(tracking_params.keys())
    filtering_params = config.getFilteringParams()
    [basic_params, labels, default_id] = config.getBasicParams(trackers, xv_input_found, extended_db)

    if len(sys.argv) > 1:
        extended_db = True
        batch_mode = True
        processArguments(sys.argv[1:])
        if avg_filename is None or agg_filename is None:
            print 'Disabling GUI and using default/command line arguments'
        else:
            print 'avg_filename=', avg_filename

    app = StandaloneTrackingApp(init_frame, db_root_path, basic_params, tracking_params,
                                filtering_params, labels, default_id, frame_buffer_size,
                                success_threshold, batch_mode=batch_mode,
                                agg_filename=agg_filename, avg_filename=avg_filename,
                                anim_app=None, extended_db=extended_db,
                                write_tracking_data=write_tracking_data,
                                tracking_data_fname=tracking_data_fname, camera_id = camera_id)
    use_plot = 1
    if batch_mode:
        app.run(show_img)
    elif use_plot:
        fig = plt.figure(0)
        fig.canvas.set_window_title('Press shift to switch to FPS')
        cid = fig.canvas.mpl_connect('key_press_event', app.onKeyPress)
        ax = plt.axes(xlim=(0, app.no_of_frames), ylim=(0, 5))
        plt.xlabel('Frame')
        plt.ylabel('Error')
        plt.title('Tracking Error')
        plt.grid(True)
        line1, line2 = ax.plot([], [], 'r', [], [], 'g')
        plt.legend(('Average', 'Current'))
        # plt.draw()
        def initPlot():
            line1.set_data([], [])
            line2.set_data([], [])
            # line3.set_data(0, 0)
            return line1, line2  #

        anim = animation.FuncAnimation(fig, app.animate, simData, init_func=initPlot,
                                       interval=0, blit=True)
        plt.show()
    else:
        i = 0
        line1 = 0
        line2 = 0
        while True:
            app.animate(i)
            i += 1

