import cv2
import sys
import time
import os, shutil
import platform
import numpy as np
from datetime import datetime

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
    'fps': 20,
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
    'out_resize_factor': 1.0,
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
out_resize_factor = params['out_resize_factor']

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

time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
out_name = 'screen_capture_{}.{}'.format(time_stamp, ext)
fourcc = cv2.VideoWriter_fourcc(*codec)

out_w, out_h = int(w*out_resize_factor), int(h*out_resize_factor)
video_out = cv2.VideoWriter(out_name, fourcc, fps, (out_w, out_h))

if video_out is None:
    raise IOError('Output video file could not be opened: {}'.format(out_name))
print('Saving {}x{} output video to {}'.format(out_w, out_h, out_name))

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

        if out_resize_factor != 1:
            out_image = resizeAR(image, resize_factor=out_resize_factor)
        else:
            out_image = image

        video_out.write(out_image)

        vid_end_t = time.time()

        grab_fps = 1.0 / float(grab_end_t - _start_t)
        proc_fps = 1.0 / float(proc_end_t - _start_t)
        vid_fps = 1.0 / float(vid_end_t - _start_t)


        if show_img:
            image = resizeAR(image, width, height,
                             resize_factor=vis_resize_factor, placement_type=1)

            cv2.imshow('screenshot', image)
            k = cv2.waitKey(1 - pause_after_frame) & 0xFF
            if k == ord('q') or k == 27:
                break
            elif k == 32:
                pause_after_frame = 1 - pause_after_frame

        sys.stdout.write('\rgrab: {} proc: {} vid: {}'.format(
            grab_fps, proc_fps, vid_fps))
        sys.stdout.flush()

    sys.stdout.write('\n\n')
    sys.stdout.flush()

    video_out.release()

    if show_img:
        cv2.destroyWindow('screenshot')
