import numpy as np
import keyboard
import sys
import os
import cv2
import ctypes

from threading import Event

interrupt_wait = Event()

from Misc import processArguments, sortKey, resizeAR, addBorder

try:
    win_wallpaper_func = ctypes.windll.user32.SystemParametersInfoA
    orig_wp_fname = ctypes.create_string_buffer(500)
    SPI_GETDESKWALLPAPER = 0x0073
    SPI_SETDESKWALLPAPER = 20

    orig_wp_fname_res = win_wallpaper_func(SPI_GETDESKWALLPAPER, 500, orig_wp_fname, 0)

    orig_wp_fname = orig_wp_fname.value.decode("utf-8")
    print("orig_wp_fname: {}".format(orig_wp_fname))

    win_wallpaper_func = ctypes.windll.user32.SystemParametersInfoW

except BaseException as e:
    raise SystemError('Wallpaper functionality unavailable: {}'.format(e))


params = {
    'src_path': '.',
    'img_ext': 'jpg',
    'show_img': 1,
    'del_src': 0,
    'start_id': 0,
    'mode': 1,
    'res': '',
    'interval': 0.05,
    'wallpaper_dir': 'log',
}

processArguments(sys.argv[1:], params)
_src_path = params['src_path']
img_ext = params['img_ext']
show_img = params['show_img']
del_src = params['del_src']
start_id = params['start_id']
mode = params['mode']
res = params['res']
wallpaper_dir = params['wallpaper_dir']
interval = params['interval']

vid_exts = ['.mkv', '.mp4', '.avi', '.mjpg', '.wmv']

print('Reading source videos from: {}'.format(_src_path))
vid_exts = ['.mkv', '.mp4', '.avi', '.mjpg', '.wmv']

if os.path.isdir(_src_path):
    src_files = [os.path.join(_src_path, k) for k in os.listdir(_src_path) for _ext in vid_exts if
                 k.endswith(_ext)]
    n_videos = len(src_files)
    if n_videos <= 0:
        raise SystemError('No input videos found')
    print('n_videos: {}'.format(n_videos))
    src_files.sort(key=sortKey)
else:
    src_files = [_src_path]

if not os.path.isdir(wallpaper_dir):
    os.makedirs(wallpaper_dir)
src_id = 0
n_sources = len(src_files)

exit_program = 0
def exit_callback():
    global exit_program
    print('Exiting')
    exit_program = 1
    interrupt_wait.set()


keyboard.add_hotkey('ctrl+alt+esc', exit_callback)

while src_id < n_sources:
    src_path = src_files[src_id]
    src_path = os.path.abspath(src_path)
    seq_name = os.path.splitext(os.path.basename(src_path))[0]

    cap = cv2.VideoCapture()
    if not cap.open(src_path):
        raise IOError('The video file ' + src_path + ' could not be opened')

    if cv2.__version__.startswith('3'):
        cv_prop = cv2.CAP_PROP_FRAME_COUNT
        h_prop = cv2.CAP_PROP_FRAME_HEIGHT
        w_prop = cv2.CAP_PROP_FRAME_WIDTH
    else:
        cv_prop = cv2.cv.CAP_PROP_FRAME_COUNT
        h_prop = cv2.cv.CAP_PROP_FRAME_HEIGHT
        w_prop = cv2.cv.CAP_PROP_FRAME_WIDTH

    total_frames = int(cap.get(cv_prop))
    _height = int(cap.get(h_prop))
    _width = int(cap.get(w_prop))

    if start_id > 0:
        print('Starting from frame_id {}'.format(start_id))

    frame_id = 0
    pause_after_frame = 0
    frames = []
    screen_size = (1920, 1080)
    while True:

        ret, src_img = cap.read()
        if not ret:
            print('\nFrame {:d} could not be read'.format(frame_id + 1))
            break

        frame_id += 1

        if frame_id <= start_id:
            continue

        user32 = ctypes.windll.user32
        screensize = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)

        img_h, img_w = src_img.shape[:2]
        wallpaper_size = (img_w, img_h)

        # wp_width = 1920
        #
        # if mode == 1:
        #     if screensize[0] == 1920 and screensize[1] == 1080:
        #         wp_border = 30
        #     else:
        #         wp_border = 0
        #     wp_height = 1080
        #     wp_start_row = screensize[1] - 1080
        #     wp_start_col = 0
        # else:
        #     wp_border = 30
        #     wp_height = screensize[1]
        #     wp_start_row = 0
        #     if screensize[0] >= 3840:
        #         wp_start_col = 1920
        #     else:
        #         wp_start_col = 0
        #
        # src_img_desktop = resizeAR(src_img, wp_width, wp_height)
        # wp_end_col = wp_start_col + src_img_desktop.shape[1]
        # wp_end_row = wp_start_row + src_img_desktop.shape[0]
        #
        #
        # src_img_desktop_full = np.zeros((screensize[1], screensize[0], 3), dtype=np.uint8)
        # src_img_desktop_full[wp_start_row:wp_end_row, wp_start_col:wp_end_col, :] = src_img_desktop

        if wallpaper_size != screensize:
            src_img = resizeAR(src_img, screensize[0], screensize[1])

        wp_fname = os.path.join(wallpaper_dir, 'wallpaper_{}.jpg'.format(frame_id))
        # wp_fname = os.path.join(wallpaper_dir, 'wallpaper.bmp')
        cv2.imwrite(wp_fname, src_img)

        # print('screensize: {}'.format(screensize))
        # print('wp_fname: {}'.format(wp_fname))

        win_wallpaper_func(SPI_SETDESKWALLPAPER, 0, wp_fname, 0)

        sys.stdout.write('\rDone {:d} frames '.format(frame_id - start_id))
        sys.stdout.flush()

        interrupt_wait.wait(interval)
        if exit_program:
            break
    if exit_program:
        break
    src_id += 1
    if src_id >= n_sources:
        src_id = 0

sys.stdout.write('\n')
sys.stdout.flush()

win_wallpaper_func(SPI_SETDESKWALLPAPER, 0, orig_wp_fname, 0)
