import cv2
import sys
import time
import os, shutil
from datetime import datetime
from pprint import pformat

from Misc import processArguments, sortKey, resizeAR

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
}

processArguments(sys.argv[1:], params)
src_path = params['src_path']
save_path = params['save_path']
img_ext = params['img_ext']
show_img = params['show_img']
del_src = params['del_src']
start_id = params['start_id']
n_frames = params['n_frames']
_width = params['width']
_height = params['height']
fps = params['fps']
codec = params['codec']
ext = params['ext']
out_postfix = params['out_postfix']
reverse = params['reverse']
save_root_dir = params['save_root_dir']

img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff', '.webp')

existing_images = {}
_pause = 0
while True:
    _src_files = [k for k in os.listdir(src_path) if
                  os.path.splitext(k.lower())[1] in img_exts]
    for _src_file in _src_files:
        _src_path = os.path.join(src_path, _src_file)
        # mod_time = os.path.getmtime(_src_path)
        _src_file_no_ext = os.path.splitext(_src_file)[0]
        try:
            _src_file_id, _src_file_timestamp = _src_file_no_ext.split('___')
        except ValueError as e:
            print('Invalid _src_file: {} with _src_file_no_ext: {}'.format(
                _src_file, _src_file_no_ext))

        if _src_file_id not in existing_images or existing_images[_src_file_id] != _src_file_timestamp:
            existing_images[_src_file] = _src_file_timestamp
            print('reading {} with time: {}'.format(_src_file_id, _src_file_timestamp))

            img = cv2.imread(_src_path)

            # out_path = _src_path+'.done'
            # print('writing to {}'.format(out_path))
            # with open(out_path, 'w') as fid:
            #     fid.write('siif')
            #     fid.close()

            os.remove(_src_path)
            cv2.imshow(_src_file_id, img)

        del_images = []
        for existing_image in existing_images.keys():
            if existing_image not in _src_files:
                cv2.destroyWindow(existing_image)
                del_images.append(existing_image)

        for del_image in del_images:
            del existing_images[del_image]

    k = cv2.waitKey(1 - _pause)
    if k == 27:
        break
    elif k == 32:
        _pause = 1 - _pause

for existing_image in existing_images.keys():
    cv2.destroyWindow(existing_image)




