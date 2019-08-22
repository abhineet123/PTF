import numpy as np
import cv2
import os
from Misc import sortKey

class ImageSequenceCapture:
    """
    :param str src_path
    :param int recursive
    """

    def __init__(self, src_path='', recursive=0, img_exts=(), logger=None):
        self.src_path = ''
        self.src_fmt = ''
        self.recursive = 0
        self.img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff')
        self.src_files = []
        self.n_src_files = 0
        self.is_open = False
        self.frame_id = 0

        if src_path:
            if self.open(src_path, recursive, img_exts):
                self.is_open = True

    def isOpened(self, cv_prop):
        return self.is_open

    def read(self):
        if self.frame_id >= self.n_src_files:
            raise IOError('Invalid frame_id: {} for sequence with {} frames'.format(
                self.frame_id, self.n_src_files
            ))
        frame = cv2.imread(self.src_files[self.frame_id])
        self.frame_id += 1
        return True, frame

    def set(self, cv_prop, _id):
        if cv_prop == cv2.CAP_PROP_POS_FRAMES:
            print('Setting frame_id to : {}'.format(_id))
            self.frame_id = _id

    def get(self, cv_prop):
        if cv_prop == cv2.CAP_PROP_POS_FRAMES:
            return self.frame_id

    def open(self, src_path='', recursive=0, img_exts=()):
        if src_path:
            img_ext = os.path.splitext(os.path.basename(src_path))[1]
            if img_ext:
                self.src_path = os.path.dirname(src_path)
                self.src_fmt = os.path.basename(src_path)
                self.img_exts = (img_ext,)
            else:
                self.src_path = src_path

            self.recursive = recursive
        if img_exts:
            self.img_exts = img_exts

        if recursive:
            src_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                             os.path.splitext(f.lower())[1] in self.img_exts]
                            for (dirpath, dirnames, filenames) in os.walk(self.src_path, followlinks=True)]
            _src_files = [item for sublist in src_file_gen for item in sublist]
        else:
            _src_files = [os.path.join(self.src_path, k) for k in os.listdir(self.src_path) if
                          os.path.splitext(k.lower())[1] in self.img_exts]

        if not _src_files:
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
