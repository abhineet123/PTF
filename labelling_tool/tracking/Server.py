import numpy as np
import argparse
import os
import socket, paramiko
import time
import cv2
import threading
import sys
import multiprocessing
import pandas as pd

import logging


def profile(self, message, *args, **kws):
    if self.isEnabledFor(PROFILE_LEVEL_NUM):
        self._log(PROFILE_LEVEL_NUM, message, args, **kws)


# from .PatchTracker import PatchTracker, PatchTrackerParams
# from .Visualizer import Visualizer, VisualizerParams
# from .Utilities import processArguments, addParamsToParser, processArgsFromParser, \
#     str2list, list2str, drawRegion
from .utils.netio import send_msg_to_connection, recv_from_connection

from libs.frames_readers import get_frames_reader
from libs.netio import bindToPort

sys.path.append('../..')

class ServerParams:
    """
    :type mode: int
    :type load_path: str
    :type continue_training: int | bool
    :type gate: GateParams
    :type patch_tracker: PatchTrackerParams
    :type visualizer: VisualizerParams
    """

    def __init__(self):
        self.cfg = 'cfg/params.cfg'
        self.mode = 0
        self.wait_timeout = 3
        self.port = 3002
        self.verbose = 0
        self.save_as_bin = 0

        self.remote_path = '/home/abhineet/acamp_code_non_root/labelling_tool/tracking'
        self.remote_cfg = 'params.cfg'
        self.remote_img_root_path = '/home/abhineet/acamp/object_detection/videos'
        self.hostname = ''
        self.username = ''
        self.password = ''

        self.img_path = ''
        self.img_paths = ''
        self.root_dir = ''
        self.save_dir = 'log'
        self.save_csv = 0
        self.track_init_frame = 1

        self.roi = ''
        self.id_number = 0
        self.init_frame_id = 0
        self.end_frame_id = -1
        self.init_bbox = ''

        # self.patch_tracker = PatchTrackerParams()
        # self.visualizer = VisualizerParams()
        self.help = {
            'cfg': 'optional ASCII text file from where parameter values can be read;'
                   'command line parameter values will override the values in this file',
            'mode': 'mode in which to run the server:'
                    ' 0: local execution'
                    ' 1: remote execution'
                    ' 2: output to terminal / GUI in local execution mode (non-server)',
            'port': 'port on which the server listens for requests',
            'save_as_bin': 'save images as binary files for faster reloading (may take a lot of disk space)',
            'img_path': 'single sequence on which patch tracker is to be run (mode=2); overriden by img_path',
            'img_paths': 'list of sequences on which patch tracker is to be run (mode=2); overrides img_path',
            'root_dir': 'optional root directory containing sequences on which patch tracker is to be run (mode=2)',

            'verbose': 'show detailed diagnostic messages',
            'patch_tracker': 'parameters for the patch tracker module',
            'visualizer': 'parameters for the visualizer module',
        }


class Server:
    """
    :type params: ServerParams
    :type logger: logging.RootLogger
    """

    def __init__(self, params, _logger):
        """
        :type params: ServerParams
        :type _logger: logging.RootLogger
        :rtype: None
        """

        self.params = params
        self.logger = _logger

        self.request_dict = {}
        self.request_list = []

        self.current_path = None
        self.frames_reader = None
        self.trainer = None
        self.tester = None
        self.visualizer = None
        self.enable_visualization = False
        self.traj_data = []

        self.trained_target = None
        self.tracking_res = None
        self.index_to_name_map = None

        self.max_frame_id = -1
        self.frame_id = -1

        self.pid = os.getpid()

        self.request_lock = threading.Lock()

        # create parsers for real time parameter manipulation
        # self.parser = argparse.ArgumentParser()
        # addParamsToParser(self.parser, self.params)

        self.client = None
        self.channel = None
        self._stdout = None
        self.remote_output = None

        if self.params.mode == 0:
            self.logger.info('Running in local execution mode')
        elif self.params.mode == 1:
            self.logger.info('Running in remote execution mode')
            self.connectToExecutionServer()
        elif self.params.mode == 2:
            self.logger.info('Running patch tracker directly')

        # self.patch_tracking_results = []

    def connectToExecutionServer(self):
        self.logger.info('Executing on {}@{}'.format(self.params.username, self.params.hostname))

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self.client.connect(
            self.params.hostname,
            username=self.params.username,
            password=self.params.password
        )
        self.channel = self.client.invoke_shell(width=1000, height=3000)
        self._stdout = self.channel.makefile()
        # self.flushChannel()

    def parseParams(self, parser, cmd_args):
        args_in = []
        # check for a custom cfg file specified at command line
        # prefix = '{:s}.'.format(name)
        if len(cmd_args) > 0 and '--cfg' in cmd_args[0]:
            _, arg_val = cmd_args[0].split('=')
            self.params.cfg = arg_val
            # print('Reading {:s} parameters from {:s}'.format(name, cfg))
        if os.path.isfile(self.params.cfg):
            file_args = open(self.params.cfg, 'r').readlines()
            # lines starting with # in the cfg file are regarded as comments and thus ignored
            file_args = ['--{:s}'.format(arg.strip()) for arg in file_args if arg.strip() and not arg.startswith('#')]
            # print('file_args', file_args)
            args_in += file_args
        # command line arguments override those in the cfg file
        args_in += ['--{:s}'.format(arg[2:]) for arg in cmd_args]
        # args_in = [arg[len(prefix):] for arg in args_in if prefix in arg]
        # print('args_in', args_in)
        args = parser.parse_args(args_in)
        processArgsFromParser(self.params, args)

    def getRemoteOutput(self):
        self.remote_output = self._stdout.readline().replace("^C", "")

    def flushChannel(self):
        # while not self.channel.exit_status_ready():
        while True:
            # if not self.channel.recv_ready():
            #     continue

            # remote_output = self._stdout.readline().replace("^C", "")

            self.remote_output = None

            p = multiprocessing.Process(target=self.getRemoteOutput)
            p.start()
            # Wait for 1 second or until process finishes
            p.join(self.params.wait_timeout)

            if p.is_alive():
                p.terminate()
                p.join()

            if not self.remote_output:
                break

            # print('remote_output: ', remote_output)
            if not self.remote_output.startswith('###'):
                sys.stdout.write(self.remote_output)
                sys.stdout.flush()

    def visualize(self, request):
        request_path = request["path"]
        csv_path = request["csv_path"]
        class_dict = request["class_dict"]
        request_roi = request["roi"]
        init_frame_id = request["frame_number"]

        save_fname_templ = os.path.splitext(os.path.basename(request_path))[0]

        df = pd.read_csv(csv_path)

        if request_path != self.current_path:
            self.frames_reader = get_frames_reader(request_path, save_as_bin=self.params.save_as_bin)
            if request_roi is not None:
                self.frames_reader.setROI(request_roi)
            self.current_path = request_path
        class_labels = dict((v, k) for k, v in class_dict.items())

        # print('self.params.visualizer.save: ', self.params.visualizer.save)
        visualizer = Visualizer(self.params.visualizer, self.logger, class_labels)
        init_frame = self.frames_reader.get_frame(init_frame_id)

        height, width, _ = init_frame.shape
        frame_size = width, height
        visualizer.initialize(save_fname_templ, frame_size)

        n_frames = self.frames_reader.num_frames
        for frame_id in range(init_frame_id, n_frames):
            try:
                curr_frame = self.frames_reader.get_frame(frame_id)
            except IOError as e:
                print('{}'.format(e))
                break

            file_path = self.frames_reader.get_file_path()
            if file_path is None:
                print('Visualization is only supported on image sequence data')
                return

            filename = os.path.basename(file_path)

            multiple_instance = df.loc[df['filename'] == filename]
            # Total # of object instances in a file
            no_instances = len(multiple_instance.index)
            # Remove from df (avoids duplication)
            df = df.drop(multiple_instance.index[:no_instances])

            frame_data = []

            for instance in range(0, len(multiple_instance.index)):
                target_id = multiple_instance.iloc[instance].loc['target_id']
                xmin = multiple_instance.iloc[instance].loc['xmin']
                ymin = multiple_instance.iloc[instance].loc['ymin']
                xmax = multiple_instance.iloc[instance].loc['xmax']
                ymax = multiple_instance.iloc[instance].loc['ymax']
                class_name = multiple_instance.iloc[instance].loc['class']
                class_id = class_dict[class_name]

                width = xmax - xmin
                height = ymax - ymin

                frame_data.append([frame_id, target_id, xmin, ymin, width, height, class_id])

            frame_data = np.asarray(frame_data)
            if not visualizer.update(frame_id, curr_frame, frame_data):
                break

        visualizer.close()

    def patchTracking(self, request=None, img_path=''):

        request_path = request["path"]
        request_roi = request["roi"]
        id_number = request['id_number']
        # init_frame_id = request["frame_number"]
        init_frame_id = 0
        init_bbox = request["bbox"]
        label = request['label']
        request_port = request["port"]

        if request_path != self.current_path:
            self.frames_reader = get_frames_reader(request_path, save_as_bin=self.params.save_as_bin)
            if request_roi is not None:
                self.frames_reader.setROI(request_roi)
            self.current_path = request_path

        n_frames = self.frames_reader.num_frames

        if self.params.end_frame_id >= init_frame_id:
            end_frame_id = self.params.end_frame_id
        else:
            end_frame_id = n_frames - 1

        for frame_id in range(init_frame_id, end_frame_id + 1):
            try:
                curr_frame = self.frames_reader.get_frame(frame_id)
            except IOError as e:
                print('{}'.format(e))
                break

            if request_port is not None:

                self.send(curr_frame, init_bbox, label, request_path, frame_id,
                          id_number, request_port)
            # self.single_object_tracking_results.append(tracking_result)

        sys.stdout.write('Closing tracker...\n')
        sys.stdout.flush()

    def send(self, curr_frame, out_bbox, label, request_path, frame_id, id_number,
             request_port, masks=None):

        # print('frame_id: {}, out_bbox: {}'.format(frame_id, out_bbox))

        if len(curr_frame.shape) == 3:
            height, width, channels = curr_frame.shape
        else:
            height, width = curr_frame.shape
            channels = 1

        tracking_result = dict(
            action="add_bboxes",
            path=request_path,
            frame_number=frame_id,
            width=width,
            height=height,
            channel=channels,
            bboxes=[out_bbox],
            scores=[0],
            labels=[label],
            id_numbers=[id_number],
            bbox_source="single_object_tracker",
            last_frame_number=frame_id - 1,
            trigger_tracking_request=False,
            num_frames=1,
            # port=request_port,
        )
        if masks is not None:
            tracking_result['masks'] = [masks, ]

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', request_port))
        send_msg_to_connection(tracking_result, sock)
        sock.close()

    def run(self):
        if self.params.mode == 2:
            img_paths = self.params.img_paths
            root_dir = self.params.root_dir

            if img_paths:
                if os.path.isfile(img_paths):
                    img_paths = [x.strip() for x in open(img_paths).readlines() if x.strip()]
                else:
                    img_paths = img_paths.split(',')
                if root_dir:
                    img_paths = [os.path.join(root_dir, name) for name in img_paths]

            elif root_dir:
                img_paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                             os.path.isdir(os.path.join(root_dir, name))]
                img_paths.sort(key=sortKey)

            else:
                img_paths = (self.params.img_path,)

            print('Running patch tracker on {} sequences'.format(len(img_paths)))
            for img_path in img_paths:
                self.patchTracking(img_path=img_path)
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bindToPort(sock, self.params.port, 'tracking')
        sock.listen(1)
        self.logger.info('Tracking server started')
        # if self.params.mode == 0:
        #     self.logger.info('Started tracking server in local execution mode')
        # else:
        #     self.logger.info('Started tracking server in remote execution mode')
        while True:
            try:
                connection, addr = sock.accept()
                connection.settimeout(None)
                msg = recv_from_connection(connection)
                connection.close()
                if isinstance(msg, list):
                    raw_requests = msg
                else:
                    raw_requests = [msg]
                for request in raw_requests:
                    # print('request: ', request)
                    request_type = request['request_type']
                    if request_type == 'patch_tracking':
                        # self.params.processArguments()
                        try:
                            self.patchTracking(request)
                        except KeyboardInterrupt:
                            continue
                    # elif request_type == 'stop':
                    #     break
                    elif request_type == 'visualize':
                        self.visualize(request)
                    else:
                        self.logger.error('Invalid request type: {}'.format(request_type))
            except KeyboardInterrupt:
                print('Exiting due to KeyboardInterrupt')
                if self.client is not None:
                    self.client.close()
                return
            except SystemExit:
                if self.client is not None:
                    self.client.close()
                return
        # self.logger.info('Stopped tracking server')

    # def run(self):
    # threading.Thread(target=self.request_loop).start()


if __name__ == '__main__':
    # get parameters
    _params = ServerParams()
    processArguments(_params, description='Tracking Server')

    # setup logger
    PROFILE_LEVEL_NUM = 9
    logging.addLevelName(PROFILE_LEVEL_NUM, "PROFILE")
    logging.Logger.profile = profile

    logging_fmt = '%(levelname)s::%(module)s::%(funcName)s::%(lineno)s :  %(message)s'
    logging_level = logging.INFO
    # logging_level = logging.DEBUG
    # logging_level = PROFILE_LEVEL_NUM
    logging.basicConfig(level=logging_level, format=logging_fmt)
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)

    server = Server(_params, _logger)
    server.run()
