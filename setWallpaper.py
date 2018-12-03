import numpy as np
import keyboard
import sys
import os
import cv2
import ctypes
from threading import Event

interrupt_wait = Event()

from Misc import processArguments, sortKey

params = {
    'src_path': ['.'],
    'save_path': '',
    'img_ext': 'jpg',
    'show_img': 1,
    'del_src': 0,
    'start_id': 0,
    'n_frames': 0,
    'transition_interval': 10,
    'random_mode': 1,
    'recursive': 1,
    'width': 0,
    'height': 0,
    'fps': 30,
    'codec': 'H264',
    'ext': 'mkv',
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
recursive = params['recursive']
random_mode = params['random_mode']
transition_interval = params['transition_interval']

MAX_TRANSITION_INTERVAL = 1000
MIN_TRANSITION_INTERVAL = 1

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
src_file_list = []

old_transition_interval = transition_interval

for src_path in _src_path:
    img_id = 0
    if os.path.isdir(src_path):
        src_dir = src_path
        img_fname = None
    elif os.path.isfile(src_path):
        src_dir = os.path.dirname(src_path)
        img_fname = src_path
    else:
        raise IOError('Invalid source path: {}'.format(src_path))

    print('Reading source images from: {}'.format(src_dir))

    img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff', '.gif')

    if recursive:
        src_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                         os.path.splitext(f.lower())[1] in img_exts]
                        for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)]
        src_file_list += [item for sublist in src_file_gen for item in sublist]
    else:
        src_file_list += [os.path.join(src_dir, k) for k in os.listdir(src_dir) if
                          os.path.splitext(k.lower())[1] in img_exts]

total_frames = len(src_file_list)
if total_frames <= 0:
    raise SystemError('No input frames found')
print('total_frames: {}'.format(total_frames))

try:
    # nums = int(os.path.splitext(img_fname)[0].split('_')[-1])
    src_file_list.sort(key=sortKey)
except:
    src_file_list.sort()

if img_fname is None:
    img_fname = src_file_list[img_id]

img_id = src_file_list.index(img_fname)

src_img = cv2.imread(img_fname)
img_h, img_w = src_img.shape[:2]
wallpaper_size = (img_w, img_h)

print("wallpaper_size: {}".format(wallpaper_size))

exit_program = 0


def loadImage(diff=0):
    global img_id, src_file_list_rand
    img_id += diff
    if img_id >= total_frames:
        img_id -= total_frames
        if random_mode:
            print('Resetting randomisation')
            src_file_list_rand = list(np.random.permutation(src_file_list))
    if img_id < 0:
        img_id = 0

    if random_mode:
        src_img_fname = src_file_list_rand[img_id]
    else:
        src_img_fname = src_file_list[img_id]

    src_img_fname = os.path.abspath(src_img_fname)

    user32 = ctypes.windll.user32
    screen_size = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)

    if wallpaper_size != screen_size:
        print('Mismatch detected between wallpaper size: {} and screen size: {}'.format(wallpaper_size, screen_size))
        exit_callback()
        return

    # print('src_img_fname: {}'.format(src_img_fname))
    win_wallpaper_func(SPI_SETDESKWALLPAPER, 0, src_img_fname, 0)


def inc_callback():
    global transition_interval, img_id
    transition_interval += 1
    print('Setting transition interval to: {}'.format(transition_interval))
    img_id -= 1
    interrupt_wait.set()


def dec_callback():
    global transition_interval, img_id
    transition_interval -= 1
    if transition_interval < 1:
        transition_interval = 1
    print('Setting transition interval to: {}'.format(transition_interval))
    img_id -= 1
    interrupt_wait.set()


def inc_callback2():
    global transition_interval, old_transition_interval, img_id
    if transition_interval == MAX_TRANSITION_INTERVAL:
        transition_interval = old_transition_interval
    else:
        old_transition_interval = transition_interval
        transition_interval = MAX_TRANSITION_INTERVAL
    print('Setting transition interval to: {}'.format(transition_interval))
    img_id -= 1
    interrupt_wait.set()


def dec_callback2():
    global transition_interval, old_transition_interval, img_id

    if transition_interval == MIN_TRANSITION_INTERVAL:
        transition_interval = old_transition_interval
    else:
        old_transition_interval = transition_interval
        transition_interval = MIN_TRANSITION_INTERVAL

    print('Setting transition interval to: {}'.format(transition_interval))
    img_id -= 1
    interrupt_wait.set()


def exit_callback():
    global exit_program
    print('Exiting')
    exit_program = 1
    interrupt_wait.set()


def next_callback():
    loadImage(1)


def prev_callback():
    loadImage(-1)


keyboard.add_hotkey('ctrl+alt+esc', exit_callback)
keyboard.add_hotkey('ctrl+alt+right', next_callback)
keyboard.add_hotkey('ctrl+alt+left', prev_callback)
keyboard.add_hotkey('ctrl+alt+up', inc_callback)
keyboard.add_hotkey('ctrl+alt+down', dec_callback)
keyboard.add_hotkey('ctrl+alt+shift+up', inc_callback2)
keyboard.add_hotkey('ctrl+alt+shift+down', dec_callback2)
if random_mode:
    print('Random mode enabled')
    src_file_list_rand = list(np.random.permutation(src_file_list))

img_id -= 1
while not exit_program:
    # print('img_id: {}'.format(img_id))
    loadImage(1)
    interrupt_wait.wait(transition_interval)
    # time.sleep(transition_interval)
    if exit_program:
        break
    interrupt_wait.clear()

win_wallpaper_func(SPI_SETDESKWALLPAPER, 0, orig_wp_fname, 0)
