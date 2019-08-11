import cv2
import sys
import time
import os, shutil
import platform
import numpy as np

if platform.system() == 'Windows':
    from mss.windows import MSS as mss
elif platform.system() == 'Linux':
    from mss.linux import MSS as mss

from mss.exception import ScreenShotError

from Misc import processArguments, sortKey, resizeAR

params = {
    'src_path': '.',
    'save_path': '',
    'img_ext': 'jpg',
    'show_img': 1,
    'del_src': 0,
    'start_id': 0,
    'n_frames': 0,
    'width': 1440,
    'height': 900,
    'fps': 30,
    # 'codec': 'FFV1',
    # 'ext': 'avi',
    'codec': 'H264',
    'ext': 'mkv',
    'out_postfix': '',
    'reverse': 0,
    # coordinates of the region from where streaming image is cropped: [left,top,right,bottom]
    'region_to_crop': [0, 0, 5760, 2160],
    'roi_resize_factor': 0.5,
    'vis_resize_factor': 0,
}

processArguments(sys.argv[1:], params)
_src_path = params['src_path']
save_path = params['save_path']
img_ext = params['img_ext']
show_img = params['show_img']
del_src = params['del_src']
start_id = params['start_id']
n_frames = params['n_frames']
width = params['width']
height = params['height']
fps = params['fps']
codec = params['codec']
ext = params['ext']
out_postfix = params['out_postfix']
reverse = params['reverse']
region_to_crop = params['region_to_crop']
roi_resize_factor = params['roi_resize_factor']
vis_resize_factor = params['vis_resize_factor']

left, top, right, bottom = region_to_crop
region_to_crop = {
    "top": top,
    "left": left,
    "width": right - left,
    "height": bottom - top
}

pause_after_frame = 1

region_to_crop = mss().monitors[0]
print('region_to_crop: {}'.format(region_to_crop))

try:
    im = mss().grab(region_to_crop)
except ScreenShotError as e:
    raise IOError('Screenshot capture failed. Please check the monitor configuration: {}'.format(e))

image = np.array(im, copy=False)
image = resizeAR(image, resize_factor=roi_resize_factor)

roi = cv2.selectROI('Select ROI', image)
print('roi: {}'.format(roi))

cv2.destroyWindow('Select ROI')

roi = [int(k / roi_resize_factor) for k in roi]

x1, y1, w, h = roi

region_to_crop['left'] += x1
region_to_crop['top'] += y1
region_to_crop['width'] = w
region_to_crop['height'] = h

print('region_to_crop: {}'.format(region_to_crop))

with mss() as sct:
    while True:
        _start_t = time.time()

        try:
            im = sct.grab(region_to_crop)
        except ScreenShotError as e:
            raise IOError('Screenshot capture failed. Please check the monitor configuration: {}'.format(e))

        grab_end_t = time.time()

        image = np.array(im, copy=False)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        proc_end_t = time.time()

        grab_fps = 1.0 / float(grab_end_t - _start_t)
        proc_fps = 1.0 / float(proc_end_t - _start_t)

        image = resizeAR(image, width, height,
                         resize_factor=vis_resize_factor, placement_type=1)

        if show_img:
            cv2.imshow('screenshot', image)
            k = cv2.waitKey(1 - pause_after_frame) & 0xFF
            if k == ord('q') or k == 27:
                break
            elif k == 32:
                pause_after_frame = 1 - pause_after_frame

        print('grab_fps: {} proc_fps: {}'.format(grab_fps, proc_fps))

    if show_img:
        cv2.destroyWindow('screenshot')
