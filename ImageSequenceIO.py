import numpy as np
import cv2
import os
from Misc import sortKey, resizeAR


class ImageSequenceWriter:
    def __init__(self, file_path, fmt='image%06d', ext='jpg', logger=None, height=0, width=0):
        self._file_path = file_path
        self._logger = logger
        self._fmt = fmt
        self._ext = ext
        self._height = height
        self._width = width

        split_path = os.path.splitext(file_path)
        self._save_dir = split_path[0]

        if not self._ext:
            try:
                self._ext = split_path[1][1:]
            except IndexError:
                self._ext = 'jpg'
            if not self._ext:
                self._ext = 'jpg'

        if not os.path.isdir(self._save_dir):
            os.makedirs(self._save_dir)
        self.frame_id = 0
        self.filename = self._fmt % self.frame_id + '.{}'.format(self._ext)
        if self._logger is None:
            self._logger = print
        else:
            self._logger = self._logger.info
        self._logger('Saving images of type {:s} to {:s}\n'.format(self._ext, self._save_dir))

    def write(self, frame, frame_id=None, prefix=''):
        if self._height or self._width:
            frame = resizeAR(frame, height=self._height, width=self._width)

        if frame_id is None:
            self.frame_id += 1
        else:
            self.frame_id = frame_id

        if prefix:
            self.filename = '{:s}.{:s}'.format(prefix, self._ext)
        else:
            self.filename = self._fmt % self.frame_id + '.{}'.format(self._ext)

        self.curr_file_path = os.path.join(self._save_dir, self.filename)
        cv2.imwrite(self.curr_file_path, frame, (cv2.IMWRITE_JPEG_QUALITY, 100))

    def release(self):
        pass


class ImageSequenceCapture:
    """
    :param str src_path
    :param int recursive
    """

    def __init__(self, src_path='', recursive=0):
        self.src_path = ''
        self.src_fmt = ''
        self.recursive = 0
        self.img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff')
        self.src_files = []
        self.n_src_files = 0
        self.is_open = False
        self.frame_id = 0

        if src_path:
            if self.open(src_path, recursive):
                self.is_open = True

    def isOpened(self, cv_prop):
        return self.is_open

    def read(self):
        if self.frame_id >= self.n_src_files:
            print('Invalid frame_id: {} for sequence with {} frames'.format(
                self.frame_id, self.n_src_files
            ))
            return False, None

        frame = cv2.imread(self.src_files[self.frame_id])
        self.frame_id += 1
        return True, frame

    def set(self, cv_prop, _id):
        if cv_prop == cv2.CAP_PROP_POS_FRAMES:
            print('Setting frame_id to : {}'.format(_id))
            self.frame_id = _id

    def get(self, cv_prop):
        if cv_prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.n_src_files
        elif cv_prop == cv2.CAP_PROP_POS_FRAMES:
            return self.frame_id
        elif cv_prop in (cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH):
            frame = cv2.imread(self.src_files[self.frame_id])
            h, w = frame.shape[:2]
            if cv_prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return h
            else:
                return w
        else:
            raise IOError('Invalid cv_prop: {}'.format(cv_prop))

    def filter_files(self, files_list):
        n_filtered_files = len(files_list)
        print('filtering {} source files to {}'.format(self.n_src_files, n_filtered_files))

        self.src_files = [k for k in self.src_files if os.path.basename(k) in files_list]
        n_src_files = len(self.src_files)

        assert n_src_files == n_filtered_files, "only {} files to be filtered found".format(n_src_files)

        self.n_src_files = n_src_files

    def open(self, src_path='', recursive=0):
        if self.is_open:
            return True

        if src_path:
            img_ext = os.path.splitext(os.path.basename(src_path))[1]
            if img_ext:
                self.src_path = os.path.dirname(src_path)
                self.src_fmt = os.path.basename(src_path)
                self.img_exts = (img_ext,)
            else:
                self.src_path = src_path

            self.recursive = recursive

        print('looking for images with formats {} in {}'.format(self.img_exts, self.src_path))

        if recursive:
            print('searching recursively')
            src_file_gen = [[os.path.join(dirpath, f) for f in filenames
                             if os.path.splitext(f.lower())[1] in self.img_exts
                             ]
                            for (dirpath, dirnames, filenames) in os.walk(self.src_path, followlinks=True)]
            _src_files = [item for sublist in src_file_gen for item in sublist]
        else:
            _src_files = [os.path.join(self.src_path, k) for k in os.listdir(self.src_path) if
                          os.path.splitext(k.lower())[1] in self.img_exts]

        n_src_files = len(_src_files)
        print('n_src_files: {}'.format(n_src_files))

        if n_src_files == 0:
            print('No images found in {}'.format(self.src_path))
            return False

        _src_files = [os.path.abspath(k) for k in _src_files]
        _src_files.sort(key=sortKey)

        self.src_files = _src_files
        self.n_src_files = len(self.src_files)

        if self.src_fmt:
            matching_files = [self.src_fmt % i for i in range(1, self.n_src_files + 1)]
            self.src_files = [k for k in self.src_files if os.path.basename(k) in matching_files]
            self.n_src_files = len(self.src_files)

        return True
