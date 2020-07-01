import cv2
import sys
import time
import ctypes
import platform
import os, shutil
from datetime import datetime
from pprint import pformat

from Misc import processArguments, sortKey, resizeAR


def main():
    params = {
        'src_path': '.',
        'save_path': '',
        'save_root_dir': '',
        'img_ext': 'jpg',
        'show_img': 1,
        'del_src': 0,
        'start_id': 0,
        'n_frames': 0,
        'width': 0,
        'height': 0,
        'fps': 30,
        # 'codec': 'FFV1',
        # 'ext': 'avi',
        'codec': 'H264',
        'ext': 'mkv',
        'out_postfix': '',
        'reverse': 0,
        'min_free_space': 30,
    }

    processArguments(sys.argv[1:], params)
    src_path = params['src_path']
    min_free_space = params['min_free_space']

    img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff', '.webp')

    src_path = os.path.abspath(src_path)
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # log_path = os.path.join(script_dir, 'siif_log.txt')
    # with open(log_path, 'w') as fid:
    #     fid.write(src_path)
    #
    # os.environ["MIIF_DUMP_IMAGE_PATH"] = src_path

    read_img_path = os.path.join(src_path, "read")

    if os.path.exists(read_img_path):
        clear_dir(read_img_path)
    else:
        os.makedirs(read_img_path)

    print('MIIF started in {}'.format(src_path))

    exit_program = 0
    while not exit_program:
        _src_files = [k for k in os.listdir(src_path) if
                      os.path.splitext(k.lower())[1] in img_exts]
        for _src_file in _src_files:
            _src_path = os.path.join(src_path, _src_file)
            _src_file_no_ext, _src_file_ext = os.path.splitext(_src_file)
            time_stamp = datetime.now().strftime("_%y%m%d_%H%M%S_%f")

            _dst_path = os.path.join(read_img_path, _src_file_no_ext + time_stamp + _src_file_ext)

            print(f'{_src_path} -> {_dst_path}')

            shutil.move(_src_path, _dst_path)

if __name__ == '__main__':
    main()
