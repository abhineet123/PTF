import subprocess
import numpy as np
import os
import sys, re

try:
    from PIL import Image, ImageChops
except ImportError as e:
    print('PIL import failed: {}'.format(e))

try:
    import cv2
except ImportError as e:
    print('OpenCV import failed: {}'.format(e))


class VideoWriterGPU:
    def __init__(self, path, fps, size):
        self._path = path
        width, height = size

        command = ['ffmpeg2',
                   '-y',
                   '-f', 'rawvideo',
                   '-codec', 'rawvideo',
                   '-s', f'{width}x{height}',  # size of one frame
                   '-pix_fmt', 'rgb24',
                   '-r', f'{fps}',  # frames per second
                   '-i', '-',  # The input comes from a pipe
                   '-an',  # Tells FFMPEG not to expect any audio
                   '-c:v', 'libx265',
                   # '-preset', 'medium',
                   '-preset', 'veryslow',
                   '-x265-params', 'lossless=1',
                   f'{self._path}']

        self._pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

        self._frame_id = 0

    def write(self, image):
        im = image[::-1].tostring()
        self._pipe.stdin.write(im)
        self._pipe.stdin.flush()

    def release(self):
        self._pipe.stdin.close()
        self._pipe.wait()

class VideoCaptureGPU:
    def __init__(self, path=None):
        if path is not None:
            self.open(path)

    def open(self, path):
        self._path = path
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "{self._path}"'
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        _data = p.stdout.read().decode("utf-8")
        print(f'_data: {_data}')

        self.width, self.height = [int(x) for x in _data.split('x')]
        print(f'width: {self.width} height: {self.height}')
        self._size = int(self.height * self.width * 3)

        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=s=x:p=0 "{self._path}"'
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        _data = p.stdout.read().decode("utf-8")
        try:
            self._n_frames = int(_data)
        except:
            # cmd = f'ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames ' \
            #     f'-of default=nokey=1:noprint_wrappers=1 {self._path}'
            # p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            # _data = p.stdout.read().decode("utf-8")
            # try:
            #     self._n_frames = int(_data)
            # except:
            #     self._n_frames = 100000
            self._n_frames = 100000

        print(f'self._n_frames: {self._n_frames}')

        cmd = ['ffmpeg',
               '-hide_banner',
               '-loglevel', 'panic',
               '-vsync', '0',
               '-hwaccel', 'nvdec',
               # '-hwaccel', 'cuvid',
               # '-c:v', 'h264_cuvid',
               '-i', f'{self._path}',
               '-f', 'image2pipe',
               '-an',
               '-vcodec', 'rawvideo',
               # '-pix_fmt', 'yuv420p ',
               # '-pix_fmt', 'rgb24',
               '-pix_fmt', 'bgr24',
               '-']
        self._pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        self._frame_id = 0
        return 1

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._n_frames

    def release(self):
        self._pipe.stdout.close()
        # self._pipe.wait()

    def read(self):
        self._frame_id += 1
        if self._frame_id > self._n_frames:
            return 0, None

        _data = self._pipe.stdout.read(self._size)
        if len(_data) == 0:
            return 0, None

        # print(f'_data: {_data}')

        # print(f'Reading frame {self._frame_id}')
        # print(f'Done')

        img = np.frombuffer(_data, dtype=np.uint8).reshape((self.height, self.width, 3))

        # cv2.imshow('img', img)
        # k = cv2.waitKey(0)
        # if k == 27:
        #     sys.exit()

        return 1, img

