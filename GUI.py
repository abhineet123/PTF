import Tkinter as tk
import sys
from TrackingParams import *
from Misc import str2num


class GUI:
    def __init__(self, obj, title):
        self.title = title
        self.params = obj.params
        self.labels = obj.labels
        self.trackers = obj.trackers

        self.obj = obj

        self.pipeline_index = obj.labels.index('pipeline')
        self.source_index = obj.labels.index('source')
        self.task_index = obj.labels.index('task')
        self.type_index = obj.labels.index('type')
        self.tracker_index = obj.labels.index('tracker')
        self.filter_index = obj.labels.index('filter')
        self.color_index = obj.labels.index('color_space')

        self.tracker_type = obj.tracker_type
        self.current_id = obj.default_id

        self.top_widgets = self.labels[:self.type_index]
        self.tracking_widgets = None
        self.divider_id = 'dataset_divider'
        self.data_widgets = [self.divider_id] + self.labels[self.type_index:]

        self.widgets = self.top_widgets + self.data_widgets

        self.param_start_id = len(self.top_widgets)

        self.parent_frames = None
        self.tracker_frames = None
        self.root = None
        self.tracker_root = None

        self.frames_created = False
        self.tracker_frames_created = False

        self.no_of_rows = 0
        self.first_instance = True
        self.first_call = True

        self.item_vars = None
        self.item_menus = None
        self.item_labels = None

        self.divide_label = None

        self.entry_labels = None
        self.entry_widgets = None
        self.entry_vars = None

        self.scv_state = False
        self.color_space = None

        self.multi_approach_ids = []

        self.start_button_label = 'Start'
        self.cancel_button_label = 'Exit'

        self.direct_capture_ids = []

        self.initWidgets()
        # self.root.after(500, obj.anim_app.startAnimation)

    def initRoot(self):
        if self.root is not None:
            self.destroyRoot()

        self.root = tk.Tk()
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/4)
        y = (hs/3)
        print 'screenwidth: ', ws
        print 'screenheight: ', hs
        self.root.geometry('+%d+%d' % (x, y))
        self.root.wm_title(self.title)
        self.root.title(self.title)
        self.frames_created = False

    def initTrackerRoot(self):
        if self.tracker_root is not None:
            self.destroyTrackerRoot()
        self.tracker_root = tk.Tk()
        self.tracker_root.wm_title('Tracking Parameters')
        self.tracker_root.title('Tracking Parameters')
        self.tracker_frames_created = False

    def removeFrames(self, frames):
        for i in xrange(1, len(frames)):
            frame = frames[i]
            frame.pack_forget()

    def removeTrackerFrames(self):
        for i in xrange(1, len(self.parent_frames)):
            frame = self.parent_frames[i]
            frame.pack_forget()
        self.frames_created = False

    def removeOptionMenus(self):
        for i in xrange(1, len(self.labels)):
            self.item_labels[i].pack_forget()
            self.item_menus[i].pack_forget()

    def getFrames(self, root, no_of_rows, make_scrollable=False):
        levels = [0]
        frames = [root]

        for node_id in xrange(no_of_rows):
            new_frame = tk.Frame(frames[node_id])
            new_frame.pack(side=tk.TOP)
            frames.append(new_frame)
            levels.append(levels[node_id] + 1)

            new_frame = tk.Frame(frames[node_id])
            new_frame.pack(side=tk.BOTTOM)
            frames.append(new_frame)
            levels.append(levels[node_id] + 1)
        frames = self.rearrangeNodes(frames, levels, no_of_rows)
        return frames

    def getTrackingWidgets(self, tracking_params, sub_id=-1):
        tracking_widgets = []
        for param_id in xrange(len(tracking_params)):
            param = tracking_params[param_id]
            if param.type == 'string_list':
                for i in xrange(len(param.val)):
                    widget = dict()
                    widget['type'] = 'string'
                    widget['name'] = param.name + ' ' + str(i)
                    widget['sub_id'] = i
                    widget['root_param'] = param
                    widget['call_back'] = self.setSubTracker
                    tracking_widgets.append(widget)
            elif param.type == 'tracking_params':
                sub_trackers = self.trackers[self.tracker_type].params['trackers'].val
                for i in xrange(len(sub_trackers)):
                    sub_tracker = sub_trackers[i]
                    if sub_tracker == 'none':
                        continue
                    sep_widget = dict()
                    sep_widget['type'] = 'separator'
                    sep_widget['name'] = sub_tracker
                    sep_widget['sub_id'] = None
                    sep_widget['root_param'] = None
                    sep_widget['call_back'] = None
                    tracking_widgets.append(sep_widget)
                    param.val[i] = TrackingParams(sub_tracker, param.list[sub_tracker])
                    sub_widgets = self.getTrackingWidgets(param.val[i].sorted_params)
                    tracking_widgets = tracking_widgets + sub_widgets
            else:
                widget = dict()
                widget['type'] = param.type
                widget['name'] = param.name
                widget['sub_id'] = sub_id
                widget['root_param'] = param
                widget['call_back'] = None
                tracking_widgets.append(widget)
        # params = list(self.trackers[self.tracker_type].params.keys())
        return tracking_widgets

    def initTrackerWidgets(self):
        # print "Creating parameter entry boxes for ", self.trackers[self.tracker_type].type, "tracker"

        # if self.entry_vars is not None:
        # self.setTrackingParams()

        print 'Starting initTrackerWidgets'

        tracking_params = self.trackers[self.tracker_type].sorted_params
        self.tracking_widgets = self.getTrackingWidgets(tracking_params)

        no_of_widgets = len(self.tracking_widgets)
        if self.tracker_root is None:
            self.initTrackerRoot()

        if self.tracker_frames_created:
            self.removeFrames(self.tracker_frames)
            self.tracker_frames_created = False

        self.tracker_frames = self.getFrames(self.tracker_root, no_of_widgets, make_scrollable=True)
        self.tracker_frames_created = True

        self.entry_labels = []
        self.entry_widgets = []
        self.entry_vars = []
        self.multi_approach_ids = []
        self.direct_capture_ids = []

        # no_of_params=len(self.trackers[self.tracker_type].params)
        frame_id = no_of_widgets
        for i in xrange(no_of_widgets):
            widget = self.tracking_widgets[i]
            # print 'widget: ', widget['name']
            # print 'sub_id: ', widget['sub_id']
            # print 'call_back: ', widget['call_back']
            root_param = widget['root_param']
            call_back = widget['call_back']
            sub_id = widget['sub_id']
            name = widget['name']

            if name == 'multi_approach':
                self.multi_approach_ids.append(i)
            if name == 'direct_capture':
                self.direct_capture_ids.append(i)

            widget_val = None
            if root_param is not None:
                if sub_id < 0:
                    widget_val = root_param.val
                else:
                    widget_val = root_param.val[sub_id]
            entry_var = None
            entry_label = None
            entry_widget = None
            if widget['type'] == 'boolean':
                callback_fnc = None
                if widget['name'] == 'enable_scv':
                    callback_fnc = self.toggleSCV
                    self.scv_state = widget_val

                entry_var = tk.IntVar(self.tracker_frames[frame_id])
                if widget_val:
                    entry_var.set(1)
                else:
                    entry_var.set(0)
                entry_widget = tk.Checkbutton(self.tracker_frames[frame_id],
                                              text=widget['name'], variable=entry_var,
                                              onvalue=1, offvalue=0,
                                              command=callback_fnc)
            elif widget['type'] == 'separator':
                entry_label = tk.Label(self.tracker_frames[frame_id],
                                       text='-------------' + widget['name'] + '-------------',
                                       relief=tk.FLAT)
            else:
                # print param.name, "=", param.val
                entry_label = tk.Label(self.tracker_frames[frame_id], text=widget['name'])
                entry_var = tk.StringVar(self.tracker_frames[frame_id])
                entry_var.set(str(widget_val))
                if widget['type'] == 'string' or widget['type'] == 'discrete':
                    if root_param.list is None:
                        entry_widget = tk.Entry(self.tracker_frames[frame_id],
                                                textvariable=entry_var, width=30)
                    else:
                        if call_back is None:
                            entry_widget = tk.OptionMenu(self.tracker_frames[frame_id],
                                                         entry_var, *root_param.list)
                        else:
                            # print 'inside::call_back=', call_back
                            entry_widget = tk.OptionMenu(self.tracker_frames[frame_id],
                                                         entry_var, *root_param.list,
                                                         command=call_back)
                elif widget['type'] == 'int' or widget['type'] == 'float':
                    entry_widget = tk.Entry(self.tracker_frames[frame_id], textvariable=entry_var, width=6)

            if entry_label is not None:
                entry_label.pack(side=tk.LEFT)
            if entry_widget is not None:
                entry_widget.pack(side=tk.LEFT)

            self.entry_labels.append(entry_label)
            self.entry_widgets.append(entry_widget)
            self.entry_vars.append(entry_var)
            frame_id += 1
            # self.setColorSpace()
        print 'Done initTrackerWidgets'

    def createOptionMenus(self):
        old_vals = None
        if self.item_vars is not None:
            old_vals = []
            for var in self.item_vars:
                old_vals.append(var.get())
        self.item_vars = []
        self.item_menus = []
        self.item_labels = []

        for i in range(len(self.labels)):
            widget = self.labels[i]
            index = self.widgets.index(widget)
            frame_id = index + self.no_of_rows
            item_var = tk.StringVar(self.parent_frames[frame_id])
            if widget == 'task':
                if old_vals is not None:
                    current_val = old_vals[i]
                else:
                    current_val = self.params[i][self.current_id[self.type_index]][self.current_id[i]]
                item_var.set(current_val)
                item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                          *self.params[i][self.current_id[self.type_index]])
            elif widget == 'source':
                if old_vals is not None:
                    current_val = old_vals[i]
                else:
                    current_val = self.params[i][self.current_id[self.pipeline_index]][self.current_id[i]]
                item_var.set(current_val)
                item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                          *self.params[i][self.current_id[self.pipeline_index]],
                                          command=self.setSource)
            else:
                if old_vals is not None:
                    current_val = old_vals[i]
                else:
                    current_val = self.params[i][self.current_id[i]]
                item_var.set(current_val)
                if widget == 'type':
                    item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                              *self.params[i], command=self.setTasks)
                elif widget == 'pipeline':
                    item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                              *self.params[i], command=self.setPipeline)
                elif widget == 'tracker':
                    item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                              *self.params[i], command=self.setTracker)
                elif widget == 'color_space':
                    self.color_space = current_val
                    item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                              *self.params[i], command=self.setColorSpace)
                elif widget == 'smoothing':
                    item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                              *self.params[i], command=self.setSmoothing)
                else:
                    item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var, *self.params[i])

            item_label = tk.Label(self.parent_frames[frame_id], text=self.labels[i], padx=10)

            item_label.pack(side=tk.LEFT)
            item_menu.pack(side=tk.LEFT)

            self.item_vars.append(item_var)
            self.item_menus.append(item_menu)
            self.item_labels.append(item_label)

    def createDivider(self):
        index = self.widgets.index(self.divider_id)
        frame_id = index + self.no_of_rows
        self.divide_label = tk.Label(self.parent_frames[frame_id], text="--------Dataset--------",
                                     relief=tk.FLAT)
        self.divide_label.pack(side=tk.LEFT)

    def createButtons(self):
        button_ok = tk.Button(self.parent_frames[-1], text=self.start_button_label,
                              command=self.ok, padx=10)
        button_ok.pack(side=tk.LEFT)
        button_cancel = tk.Button(self.parent_frames[-1], text=self.cancel_button_label,
                                  command=self.exit, padx=10)
        button_cancel.pack(side=tk.LEFT)

    def initWidgets(self, start_label=None):
        if not self.first_instance:
            self.updateCurrentID()

        if self.root is None:
            self.initRoot()

        if self.frames_created:
            self.removeFrames(self.parent_frames)
            self.frames_created = False

        if start_label is not None:
            self.start_button_label = start_label

        self.widgets = self.top_widgets + self.data_widgets
        self.no_of_rows = len(self.widgets)

        self.parent_frames = self.getFrames(self.root, self.no_of_rows)
        self.frames_created = True

        self.initTrackerWidgets()
        self.createDivider()
        self.createOptionMenus()
        self.createButtons()

        self.setSource()
        self.setTracker()
        self.setColorSpace()
        self.setSmoothing()
        self.setPipeline()

        if self.obj.auto_start:
            self.ok()
        else:
            self.root.mainloop()

    def setSubTracker(self, val):
        print 'val=', val
        self.setTrackingParams()
        # self.trackers[self.tracker_type].params['trackers'].val[tracker_id] = val
        self.initTrackerWidgets()

    def setSource(self, source=None):
        if source is None:
            if self.item_vars is None:
                return
            else:
                source = self.item_vars[self.labels.index('source')].get()

        pipeline = self.item_vars[self.pipeline_index].get()
        pipeline_id = self.params[self.pipeline_index].index(pipeline)
        # source_id = self.params[self.source_index][pipeline_id].index(source)
        for i in xrange(self.type_index, len(self.labels)):
            item_menu = self.item_menus[i]
            item_label = self.item_labels[i]
            if source == 'usb camera' or source == 'firewire camera':
                item_menu.configure(state='disabled')
                item_label.configure(state='disabled')
            else:
                item_menu.configure(state='normal')
                item_label.configure(state='normal')
        if source == 'usb camera' or source == 'firewire camera':
            self.divide_label.configure(state='disabled')
        else:
            self.divide_label.configure(state='normal')

    def toggleSCV(self):
        scv_id = self.trackers[self.tracker_type].params['enable_scv'].id
        value = self.entry_vars[scv_id].get()
        if value == 1:
            self.scv_state = True
            self.setFilterState('disabled', 'none')
        elif value == 0:
            self.scv_state = False
            if self.color_space.lower() == 'grayscale':
                self.setFilterState('normal')
        else:
            raise SystemExit('Invalid value for SCV checkbox: ' + str(value))


    def setColorSpace(self, value=None):
        if value is None:
            if self.item_vars is None:
                return
            else:
                value = self.item_vars[self.labels.index('color_space')].get()
        self.color_space = value
        if value.lower() != 'grayscale':
            # self.setFilterState('disabled', 'none')
            self.setMultiApproachState('normal')
        else:
            if not self.scv_state:
                self.setFilterState('normal')
            self.setMultiApproachState('disabled', 'none')

    def setSmoothing(self, value=None):
        if value is None:
            if self.item_vars is None:
                return
            else:
                value = self.item_vars[self.labels.index('smoothing')].get()
        if value.lower() == 'none':
            self.setSmoothingKernelState(state='disabled')
        else:
            self.setSmoothingKernelState(state='normal')


    def setTracker(self, value=None):
        if value is None:
            if self.item_vars is None:
                return
            else:
                value = self.item_vars[self.labels.index('tracker')].get()
        self.tracker_type = value
        print 'tracker_type=', self.tracker_type
        if self.tracker_type == 'xv_ssd':
            print 'setting color space to RGB'
            self.item_vars[self.labels.index('color_space')].set('RGB')
        self.initTrackerWidgets()


    def setTasks(self, value):
        type_id = self.params[self.type_index].index(value)
        frame_id = self.widgets.index('task') + self.no_of_rows
        try:
            self.item_vars[self.task_index].set(self.params[self.task_index][type_id][self.current_id[self.task_index]])
        except IndexError:
            self.current_id[self.task_index]=0
            self.item_vars[self.task_index].set(self.params[self.task_index][type_id][self.current_id[self.task_index]])
        self.item_menus[self.task_index].pack_forget()
        self.item_menus[self.task_index] = tk.OptionMenu(self.parent_frames[frame_id],
                                                         self.item_vars[self.task_index],
                                                         *self.params[self.task_index][type_id])
        self.item_menus[self.task_index].pack(side=tk.LEFT)


    def setPipeline(self, value=None):
        if value is None:
            if self.item_vars is None:
                return
            else:
                value = self.item_vars[self.labels.index('pipeline')].get()
        pipeline_id = self.params[self.pipeline_index].index(value)
        frame_id = self.widgets.index('source') + self.no_of_rows
        self.item_vars[self.source_index].set(
            self.params[self.source_index][pipeline_id][self.current_id[self.source_index]])
        self.item_menus[self.source_index].pack_forget()
        self.item_menus[self.source_index] = tk.OptionMenu(self.parent_frames[frame_id],
                                                           self.item_vars[self.source_index],
                                                           *self.params[self.source_index][pipeline_id],
                                                           command=self.setSource)
        self.item_menus[self.source_index].pack(side=tk.LEFT)
        if self.entry_vars is None or self.entry_labels is None:
            return
        # print 'direct_capture_ids: ', self.direct_capture_ids
        # print 'entry_labels: ', self.entry_labels

        for dc_id in self.direct_capture_ids:
            if value == 'XVision':
                # self.entry_vars[dc_id].set(1)
                # self.entry_labels[dc_id].configure(state='normal')
                self.entry_widgets[dc_id].configure(state='normal')
            else:
                print 'Disabling Xvision direct capture'
                self.entry_vars[dc_id].set(0)
                # self.entry_labels[dc_id].configure(state='disabled')
                self.entry_widgets[dc_id].configure(state='disabled')


    def setFilterState(self, state, val=None):

        filter_menu = self.item_menus[self.filter_index]
        filter_label = self.item_labels[self.filter_index]
        filter_var = self.item_vars[self.filter_index]

        # print 'state=', state
        filter_menu.configure(state=state)
        filter_label.configure(state=state)
        if val is not None:
            filter_var.set(val)

    def setMultiApproachState(self, state=None, val=None):
        for id in self.multi_approach_ids:
            entry_menu = self.entry_widgets[id]
            entry_label = self.entry_labels[id]
            entry_var = self.entry_vars[id]

            if state is None:
                if entry_var.get().lower() == 'none':
                    state = 'disabled'
                else:
                    state = 'normal'
            if entry_menu.cget('state') != state:
                entry_menu.configure(state=state)
            if entry_label.cget('state') != state:
                entry_label.configure(state=state)
            if val is None:
                val = self.tracking_widgets[id]['root_param'].val
            entry_var.set(val)

    def setSmoothingKernelState(self, state, val=None):

        smoothing_kernel_index = self.labels.index('smoothing_kernel')
        smoothing_kernel_menu = self.item_menus[smoothing_kernel_index]
        smoothing_kernel_label = self.item_labels[smoothing_kernel_index]
        smoothing_kernel_var = self.item_vars[smoothing_kernel_index]

        smoothing_kernel_menu.configure(state=state)
        smoothing_kernel_label.configure(state=state)
        if val is not None:
            smoothing_kernel_var.set(val)

    def ok(self):
        init_params = []
        for i in xrange(len(self.item_vars)):
            init_params.append(self.item_vars[i].get())
        self.setTrackingParams()
        if not self.obj.initSystem(init_params):
            sys.exit()

        self.destroyTrackerRoot()
        self.destroyRoot()
        if self.first_call:
            self.first_call = False
            # self.root.quit()
        else:
            self.obj.reset = True

    def exit(self):
        self.obj.exit_event = True
        if not self.first_call:
            self.obj.writeResults()
        sys.exit()

    def destroyRoot(self):
        self.root.destroy()
        self.root = None
        self.parent_frames = None

    def destroyTrackerRoot(self):
        self.tracker_root.destroy()
        self.tracker_root = None
        self.tracker_frames = None

    def updateCurrentID(self):
        # print "\n\n"
        # print "in updateCurrentID:"
        for i in xrange(len(self.item_vars)):
            current_val = self.item_vars[i].get()
            # print "updating ", self.labels[i], "to ", current_val
            if i == self.task_index:
                current_type = self.item_vars[self.type_index].get()
                current_type_id = self.params[self.type_index].index(current_type)
                current_id = self.params[i][current_type_id].index(current_val)
            elif i == self.source_index:
                current_pipeline = self.item_vars[self.pipeline_index].get()
                current_pipeline_id = self.params[self.pipeline_index].index(current_pipeline)
                current_id = self.params[i][current_pipeline_id].index(current_val)
            else:
                # if self.labels[i]=='smoothing_kernel':
                # current_val=int(current_val)
                current_id = self.params[i].index(current_val)
            self.current_id[i] = current_id
            # print "\n\n"


    def setTrackingParams(self):
        # print "\n\n"
        # print "in setTrackingParams found:"
        for i in xrange(len(self.entry_vars)):
            if self.entry_vars[i] is None:
                continue
            val_str = self.entry_vars[i].get()

            widget = self.tracking_widgets[i]
            root_param = widget['root_param']
            sub_id = widget['sub_id']
            param_type = widget['type']

            if param_type == 'int' or param_type == 'float' or param_type == 'discrete':
                val = str2num(val_str)
            elif param_type == 'string':
                val = val_str
            elif param_type == 'boolean':
                if val_str == 0:
                    val = False
                else:
                    val = True
            else:
                raise SystemExit("invalid param type: " + param_type +
                                 ' for parameter ' + widget['name'])

            if sub_id < 0:
                root_param.val = val
            else:
                root_param.val[sub_id] = val


    def shiftFromLast(self, tree, n):
        temp_node = tree[-1]
        for i in xrange(1, n + 1):
            tree[-i] = tree[-(i + 1)]
        tree[n] = temp_node
        return tree


    def shiftToLast(self, tree, n):
        temp_node = tree[n]
        for i in xrange(n, len(tree) - 1):
            tree[i] = tree[i + 1]
        tree[-1] = temp_node
        return tree


    def rearrangeNodes(self, tree, levels, nrows):
        for i in xrange(1, nrows + 1):
            if (levels[-1] > levels[nrows + i - 1]):
                tree = self.shiftFromLast(tree, nrows)
                levels = self.shiftFromLast(levels, nrows)
            else:
                break
        return tree