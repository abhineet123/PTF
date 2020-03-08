import os
import re
import cv2
import math
import sys, time, random, glob, shutil
import numpy as np
import functools
import psutil
import inspect
import keyboard

# import mouse
# from pynput import mouse

import win32gui, win32con
import win32api

# import ctypes
# from pywinauto import application

from pprint import pformat
from datetime import datetime
from threading import Event
import threading
from subprocess import Popen, PIPE
from multiprocessing import Process
import multiprocessing
import imageio
from PIL import Image

from Misc import processArguments, sortKey, stackImages, resizeAR, addBorder, trim
import sft

# from Misc import VideoCaptureGPU as VideoCapture
VideoCapture = cv2.VideoCapture


# from wand.image import Image as wandImage


def checkImage(fn):
    proc = Popen(['identify', '-verbose', fn], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out, err


def hideBorder(_win_name, on_top):
    win_handle = win32gui.FindWindow(None, _win_name)
    style = win32gui.GetWindowLong(win_handle, win32con.GWL_STYLE)
    style = style & ~win32con.WS_OVERLAPPEDWINDOW
    style = style | win32con.WS_POPUP
    if on_top:
        style = style | win32con.WS_EX_TOPMOST
        style_2 = win32con.HWND_TOPMOST
    else:
        style_2 = win32con.HWND_NOTOPMOST

    # style = style | win32con.WS_EX_TOPMOST
    # style_2 = win32con.HWND_TOPMOST

    win32gui.SetWindowLong(win_handle, win32con.GWL_STYLE, style)
    win32gui.SetWindowPos(win_handle, style_2, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)


# hotkeys_available = 0
# try:
#     import ctypes
#     from ctypes import wintypes
#     import win32con
#     byref = ctypes.byref
#     user32 = ctypes.windll.user32
#     hotkeys_available = 1
# except ImportError as e:
#     print('Hotkeys cannot be registered: {}'.format(e))
#     hotkeys_available = 0


params = {
    'src_root_dir': '.',
    'src_path': '.',
    'src_dirs': '',
    'width': 0,
    'height': 0,
    'min_height_ratio': 0.40,
    'speed': 0.5,
    'show_img': 0,
    'quality': 3,
    'resize': 0,
    'mode': 0,
    'auto_progress': 0,
    'auto_progress_video': 0,
    'max_switches': 1,
    'transition_interval': 5,
    'random_mode': 0,
    'recursive': 1,
    'fullscreen': 0,
    'reversed_pos': 1,
    'dup_reversed_pos': [],
    'double_click_interval': 0.1,
    'n_images': 1,
    'borderless': 1,
    'preserve_order': 0,
    'set_wallpaper': 0,
    'n_wallpapers': 1000,
    'wallpaper_dir': '',
    'wallpaper_mode': 0,
    'widescreen_mode': 0,
    'multi_mode': 0,
    'trim_images': 1,
    'alpha': 1.0,
    'show_window': 1,
    'enable_hotkeys': 0,
    'check_images': 0,
    'move_to_right': 0,
    'on_top': 1,
    'second_from_top': 0,
    'top_border': 0,
    'bottom_border': 0,
    'keep_borders': 0,
    'monitor_id': -1,
    'dup_monitor_ids': [],
    'win_offset_x': 0,
    'win_offset_y': 0,
    'duplicate_window': 0,
    'custom_grid_size': '',
    'reverse_video': 1,
    'images_as_video': 0,
    'frg_win_titles': [],
    'only_maximized': 1,
    'video_mode': 0,
    'lazy_video_load': 1,
    'fps': 30,
    'win_name': '',
    'other_win_name': '',
    'log_color': '',
    'parallel_read': 4,
}


def main(args, multi_exit_program=None,
         # sft_vars=None
         ):
    # is_switching = 0

    print('Here we are')
    p = psutil.Process(os.getpid())
    try:
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except AttributeError:
        os.nice(20)

    processArguments(args, params)
    src_root_dir = params['src_root_dir']
    src_path = params['src_path']
    src_dirs = params['src_dirs']
    _width = params['width']
    _height = params['height']
    min_height_ratio = params['min_height_ratio']
    speed = params['speed']
    show_img = params['show_img']
    quality = params['quality']
    resize = params['resize']
    mode = params['mode']
    move_to_right = params['move_to_right']
    widescreen_mode = params['widescreen_mode']
    auto_progress = params['auto_progress']
    auto_progress_video = params['auto_progress_video']
    max_switches = params['max_switches']
    transition_interval = params['transition_interval']
    fps = params['fps']
    random_mode = params['random_mode']
    recursive = params['recursive']
    fullscreen = params['fullscreen']
    reversed_pos = params['reversed_pos']
    dup_reversed_pos = params['dup_reversed_pos']
    double_click_interval = params['double_click_interval']
    n_images = params['n_images']
    borderless = params['borderless']
    preserve_order = params['preserve_order']
    set_wallpaper = params['set_wallpaper']
    wallpaper_dir = params['wallpaper_dir']
    wallpaper_mode = params['wallpaper_mode']
    on_top = params['on_top']
    second_from_top = params['second_from_top']
    n_wallpapers = params['n_wallpapers']
    multi_mode = params['multi_mode']
    trim_images = params['trim_images']
    alpha = params['alpha']
    show_window = params['show_window']
    enable_hotkeys = params['enable_hotkeys']
    custom_grid_size = params['custom_grid_size']
    check_images = params['check_images']
    top_border = params['top_border']
    keep_borders = params['keep_borders']
    bottom_border = params['bottom_border']
    monitor_id = params['monitor_id']
    dup_monitor_ids = params['dup_monitor_ids']
    win_offset_x = params['win_offset_x']
    win_offset_y = params['win_offset_y']
    duplicate_window = params['duplicate_window']
    reverse_video = params['reverse_video']
    images_as_video = params['images_as_video']
    frg_win_titles = params['frg_win_titles']
    only_maximized = params['only_maximized']
    video_mode = params['video_mode']
    lazy_video_load = params['lazy_video_load']
    win_name = params['win_name']
    other_win_name = params['other_win_name']
    log_color = params['log_color']
    parallel_read = params['parallel_read']

    if log_color:
        from colorlog import ColoredFormatter
        import logging
        logging_fmt = ColoredFormatter(
            '%(log_color)s%(message)s',
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': log_color,
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
        logging_level = logging.INFO
        logging.basicConfig(level=logging_level, format=logging_fmt)
        _logger = logging.getLogger()
        _logger.setLevel(logging_level)
        _logger.handlers[0].setFormatter(logging_fmt)

        def _print(*args):
            out_str = args[0]
            if len(args) > 1:
                for arg in args:
                    out_str += '{}'.format(arg)
            _logger.info(out_str)
    else:
        _print = print

    _print('args:\n{}'.format(pformat(args)))

    interrupt_wait = Event()

    win_utils_available = 1
    try:
        import winUtils

        _print('winUtils is available')
    except ImportError as e:
        win_utils_available = 0
        _print('Failed to import winUtils: {}'.format(e))
    try:
        from ctypes import windll, Structure, c_long, byref

        # Get active window id
        # https://msdn.microsoft.com/en-us/library/ms633505
        # winID = windll.user32.GetForegroundWindow()
        # print("current window ID: {}".format(winID))

        active_winID = win32gui.GetForegroundWindow()
        _print("active_winID: {}".format(active_winID))

        active_win_name = win32gui.GetWindowText(active_winID)
        _print("active_win_name: {}".format(active_win_name))

        user32 = windll.user32

        # screensize = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
        # print("screensize: {}".format(screensize))

        class POINT(Structure):
            _fields_ = [("x", c_long), ("y", c_long)]

        def queryMousePosition():
            pt = POINT()
            windll.user32.GetCursorPos(byref(pt))
            return pt

        mousePos = queryMousePosition()
        _print("mouse position x: {} y: {}".format(mousePos.x, mousePos.y))
    except ImportError as e:
        mousePos = None

    if wallpaper_mode and not set_wallpaper:
        set_wallpaper = 1

    wp_id = 0
    try:
        import ctypes

        win_wallpaper_func = ctypes.windll.user32.SystemParametersInfoA
        orig_wp_fname = ctypes.create_string_buffer(500)
        SPI_GETDESKWALLPAPER = 0x0073
        SPI_SETDESKWALLPAPER = 20

        orig_wp_fname_res = win_wallpaper_func(SPI_GETDESKWALLPAPER, 500, orig_wp_fname, 0)
        # print("orig_wp_fname_res: {}".format(orig_wp_fname_res))
        # print("orig_wp_fname raw: {}".format(orig_wp_fname.raw))
        _print("orig_wp_fname value: {}".format(orig_wp_fname.value))
        # print("orig_wp_fname: {}".format(orig_wp_fname))

        orig_wp_fname = orig_wp_fname.value.decode("utf-8")
        # orig_wp = cv2.imread(orig_wp_fname)

        win_wallpaper_func = ctypes.windll.user32.SystemParametersInfoW

    except BaseException as e:
        _print('Wallpaper functionality unavailable: {}'.format(e))
        set_wallpaper = 0

    else:
        EnumWindows = ctypes.windll.user32.EnumWindows
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        GetWindowText = ctypes.windll.user32.GetWindowTextW
        GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
        IsWindowVisible = ctypes.windll.user32.IsWindowVisible

        old_speed = speed
        speed = 0
        is_paused = 1
        monitors = [
            [0, 0],
            [-1920, 0],
            [0, -1080],
            [1920, 0],
            [1920, -1080],
        ]

        def get_monitor_id(x, y):
            monitor_id = 0
            min_dist = np.inf
            for curr_id, monitor in enumerate(monitors):
                centroid_x = (monitor[0] + monitor[0] + 1920) / 2.0
                centroid_y = (monitor[1] + monitor[1] + 1080) / 2.0
                dist = (x - centroid_x) ** 2 + (y - centroid_y) ** 2
                if dist < min_dist:
                    min_dist = dist
                    monitor_id = curr_id
            return monitor_id

        frg_titles = []
        frg_positions = []
        frg_win_borders = []
        frg_win_handles = []
        frg_reversed_pos = []
        DwmGetWindowAttribute = None

        def _exit_neatly():
            nonlocal other_win_name, second_from_top, sft_exit_program, multi_exit_program
            if multi_exit_program is not None:
                multi_exit_program.value = 1
            if other_win_name:
                try:
                    _win_handle_2 = win32gui.FindWindow(None, other_win_name)
                    win32api.PostMessage(_win_handle_2, win32con.WM_CHAR, 0x1B, 0)
                except:
                    pass

            if second_from_top:
                sft_exit_program.value = 1

        if frg_win_titles:
            titles = []
            win_pos = []
            win_border = []
            win_handles = []

            try:
                DwmGetWindowAttribute = ctypes.windll.dwmapi.DwmGetWindowAttribute
            except WindowsError:
                pass

            def findWholeWord(w):
                return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

            def foreach_window(hwnd, lParam):
                rect = win32gui.GetWindowRect(hwnd)
                # x = rect[0]
                # y = rect[1]
                # w = rect[2] - x
                # h = rect[3] - y
                # print("Window %s:" % win32gui.GetWindowText(hwnd))
                # print("\tLocation: (%d, %d)" % (x, y))
                # print("\t    Size: (%d, %d)" % (w, h))

                if IsWindowVisible(hwnd):
                    length = GetWindowTextLength(hwnd)
                    buff = ctypes.create_unicode_buffer(length + 1)
                    GetWindowText(hwnd, buff, length + 1)
                    titles.append((hwnd, buff.value))
                    win_handles.append(hwnd)

                    if DwmGetWindowAttribute:
                        ext_rect = ctypes.wintypes.RECT()
                        DWMWA_EXTENDED_FRAME_BOUNDS = 9
                        DwmGetWindowAttribute(
                            ctypes.wintypes.HWND(hwnd),
                            ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
                            ctypes.byref(ext_rect),
                            ctypes.sizeof(ext_rect)
                        )
                        border = [ext_rect.left - rect[0], ext_rect.top - rect[1],
                                  rect[2] - ext_rect.right, rect[3] - ext_rect.bottom]
                        rect = [rect[0] + border[0], rect[1] + border[1],
                                rect[2] - border[2], rect[3] - border[3]]

                        win_border.append(border)

                    win_pos.append(rect)

                return True

            win32gui.EnumWindows(foreach_window, None)

            # for i in range(len(titles)):
            #     print(titles[i])

            for frg_win_title in frg_win_titles:
                _reversed_pos = reversed_pos
                if frg_win_title.startswith('!!!'):
                    frg_win_title = frg_win_title.lstrip('!')
                    _reversed_pos = 2
                elif frg_win_title.startswith('!!'):
                    frg_win_title = frg_win_title.lstrip('!!')
                    _reversed_pos = 1
                elif frg_win_title.startswith('!'):
                    frg_win_title = frg_win_title.lstrip('!')
                    _reversed_pos = 0
                # target_id = [i for i, k in enumerate(titles) if frg_win_title in k[1]]
                # target_id = [i for i, k in enumerate(titles) if
                #              k[1].startswith(frg_win_title) or findWholeWord(frg_win_title)(k[1])]

                target_id = [i for i, k in enumerate(titles) if f' {k[1]} '.startswith(f'{frg_win_title}')]

                # target_title = [k[1] for k in titles if k[1].startswith(frg_win_titles)]
                # target_pos = [k[1] for k in win_pos if k[1].startswith(frg_win_titles)]

                if not target_id:
                    target_id = [i for i, k in enumerate(titles) if f' {frg_win_title} ' in f' {k[1]} ']

                if not target_id:
                    _print(f'\nWindow with frg_win_title {frg_win_title} not found\n')

                for _target_id in target_id:
                    frg_titles.append(titles[_target_id][1])
                    frg_positions.append(win_pos[_target_id])
                    frg_win_borders.append(win_border[_target_id])
                    frg_win_handles.append(win_handles[_target_id])
                    frg_reversed_pos.append(_reversed_pos)

                    _print(f'{frg_win_title} :: found window {frg_titles[-1]} with '
                           f'handle {frg_win_handles[-1]} and '
                           f'position: {frg_positions[-1]} '
                           f'border: {frg_win_borders[-1]}'
                           f'reversed_pos: {frg_reversed_pos[-1]}'
                           )

            frg_win_id = 0
            frg_target_title = frg_titles[frg_win_id]
            frg_target_position = frg_positions[frg_win_id]
            frg_target_win_handle = frg_win_handles[frg_win_id]

            _print('Using window: {} at {} as foreground'.format(frg_target_title, frg_target_position))

            monitor_id = get_monitor_id(frg_target_position[0], frg_target_position[1])

    sft_exceptions = ['PotPlayer', 'Free Alarm Clock', 'MPC-HC', 'DisplayFusion',
                      'GPU-Z', 'IrfanView', 'WinRAR', 'Jump List for ']

    sft_exceptions_multi = [('XY:(', ') - RGB:(', ', HTML:('), ]

    widescreen_monitor = [-1920, -1080]

    if wallpaper_mode:
        enable_hotkeys = 1

    predef_n_images = [1, 2, 3, 4, 6, 8, 9, 10, 12, 16, 18, 20, 24]
    predef_grid_sizes = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        6: (2, 3),
        8: (2, 4),
        9: (3, 3),
        10: (2, 5),
        12: (4, 3),
        16: (4, 4),
        18: (3, 6),
        20: (4, 5),
        24: (6, 4),
    }
    try:
        predef_n_image_id = predef_n_images.index(n_images)
    except ValueError:
        predef_n_image_id = 0

    n_predef_n_images = len(predef_n_images)
    if custom_grid_size:
        custom_grid_size = [int(x) for x in custom_grid_size.split('x')]
        grid_size = custom_grid_size
        if grid_size[0] == 0 and grid_size[1] == 0:
            raise IOError('Invalid custom_grid_size: {}'.format(custom_grid_size))
        if grid_size[0] == 0:
            grid_size[0] = int(math.ceil(n_images / grid_size[1]))
        elif grid_size[1] == 0:
            grid_size[1] = int(math.ceil(n_images / grid_size[0]))
        set_grid_size = 0
    else:
        grid_size = None
        set_grid_size = 1
    try:
        cv_windowed_mode_flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL
    except:
        cv_windowed_mode_flags = cv2.WINDOW_AUTOSIZE

    if monitor_id < 0 and mousePos is not None:
        monitor_id = get_monitor_id(mousePos.x, mousePos.y)
    elif monitor_id >= len(monitors):
        raise IOError('Invalid monitor_id: {}'.format(monitor_id))
    _print('monitor_id: {}'.format(monitor_id))

    if not dup_monitor_ids:
        dup_monitor_ids = [monitor_id, ]
    if duplicate_window:
        _print('dup_monitor_ids: {}'.format(dup_monitor_ids))

    if not dup_reversed_pos:
        dup_reversed_pos = []
        for _i, _ in enumerate(dup_monitor_ids):
            dup_reversed_pos.append(reversed_pos)

    if _width == 0 or _height == 0:
        if widescreen_mode:
            width = 5760
            height = 2160
        else:
            width = 1920
            if mode == 0:
                height = 1080
            else:
                height = 2160
        _width, _height = width, height
    else:
        width, height = _width, _height

    aspect_ratio = float(width) / float(height)
    direction = -1
    n_switches = 0
    start_time = end_time = 0
    src_start_row = src_start_col = src_end_row = src_end_col = 0
    row_offset = col_offset = 0

    src_img = stack_idx = stack_locations = lc_start_t = rc_start_t = None
    prev_pos = prev_win_pos = None
    end_exec = 0
    src_images = []
    img_fnames = {}

    img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff')
    vid_exts = ('.mp4', '.avi', '.mkv', '.gif')

    transition_interval_diff = 1

    video_files_list = []
    n_videos = vid_id = 0
    # video_mode = 0
    rotate_images = 0
    src_path = os.path.abspath(src_path)

    if os.path.isdir(src_path):
        src_dir = src_path
        img_fname = None
    elif os.path.isfile(src_path):
        src_dir = os.path.dirname(src_path)
        _ext = os.path.splitext(src_path)[1]
        if _ext in vid_exts:
            video_mode = 1
            img_fname = None
        else:
            img_fname = src_path
    else:
        raise IOError('Invalid source path: {}'.format(src_path))

    # def readVideoFrames(cap):
    #     global total_frames, src_file_list
    #     while True:
    #         ret, src_img = cap.read()
    #         if not ret:
    #             break
    #         src_file_list.append(src_img)
    #     total_frames = len(src_file_list)

    def read_images(_load_id, start_id, diff, _files, n_files, _img_sequences):
        for _file_id in range(start_id, n_files, diff):
            _file = _files[_file_id]
            _img_sequences[_load_id][_file] = cv2.imread(_file)

    def loadVideo(_load_id):
        nonlocal src_files, total_frames, img_id, img_sequences

        if os.path.isdir(src_path):
            # _print('Loading frames from video image sequence {}'.format(src_path))
            _src_files = [os.path.join(src_path, k) for k in os.listdir(src_path) if
                          os.path.splitext(k.lower())[1] in img_exts]
            try:
                # nums = int(os.path.splitext(img_fname)[0].split('_')[-1])
                _src_files.sort(key=img_sortKey)
            except:
                _src_files.sort()

            img_sequences[_load_id] = {}
            
            if parallel_read:
                n_files = len(_src_files)
                # n_threads = parallel_read + 1
                for _id in range(parallel_read):
                    thread = threading.Thread(target=read_images, args=(_load_id, _id, parallel_read, _src_files, n_files,
                                                                        img_sequences))
                    thread.start()

        elif os.path.isfile(src_path):
            # _print('Reading frames from video file {}'.format(src_path))

            _ext = os.path.splitext(src_path)[1]
            _src_files = []

            if _ext == '.gif':
                gif = imageio.mimread(src_path)
                meta_data = [img.meta for img in gif]
                # _print('gif meta_data: {}'.format(pformat(meta_data)))
                _src_files = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3
                              else cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                              for img in gif]
            elif _ext in img_exts:
                # if not os.path.isfile(src_path):
                #     raise IOError('Image does not exist: {}'.format(src_path))
                _src_files = [cv2.imread(src_path), ]
            else:
                # cap = cv2.VideoCapture()
                cap = VideoCapture()
                if not cap.open(src_path):
                    _exit_neatly()
                    raise IOError('The video file ' + src_path + ' could not be opened')
                if lazy_video_load:
                    _src_files = cap
                else:
                    while True:
                        ret, src_img = cap.read()
                        if not ret:
                            break
                        _src_files.append(src_img)
                        # total_frames += 1
        else:
            if other_win_name:
                _exit_neatly()
            raise IOError('Source does not exist: {}'.format(src_path))
            # return

        if isinstance(_src_files, list):
            if auto_progress and reverse_video:
                _src_files += list(reversed(_src_files))
            total_frames[_load_id] = len(_src_files)
        elif isinstance(_src_files, VideoCapture):
            if cv2.__version__.startswith('2'):
                cv_prop = cv2.cv.CAP_PROP_FRAME_COUNT
            else:
                cv_prop = cv2.CAP_PROP_FRAME_COUNT
            total_frames[_load_id] = int(cap.get(cv_prop))

        # _print('Found {} frames'.format(total_frames[_load_id]))
        src_files[_load_id] = _src_files

        img_id[_load_id] = 0

        # if cv2.__version__.startswith('3'):
        #     cv_prop = cv2.CAP_PROP_FRAME_COUNT
        # else:
        #     cv_prop = cv2.cv.CAP_PROP_FRAME_COUNT
        #
        # total_frames = int(cap.get(cv_prop))
        # readVideoFrames(cap)

        # thread = threading.Thread(target=readVideoFrames, args=(cap, ))
        # thread.start()
        # time.sleep(0.1)

    if random_mode:
        _print('Random mode enabled')

    if auto_progress:
        _print('Auto progression enabled')

    src_files = {}
    img_sequences = {}
    src_files_rand = {}
    total_frames = {}
    img_id = {}

    img_sortKey = functools.partial(sortKey, only_basename=0)

    if src_dirs:
        src_dirs = src_dirs.split(',')
        # inc_src_dirs = [k for k in src_dirs if k[0] != '!']
        # exc_src_dirs = [k for k in src_dirs if k[0] == '!']

        src_dirs = [os.path.join(src_root_dir, k) if k[0] != '!' else os.path.join('!' + src_root_dir, k[1:])
                    for k in src_dirs]

        # if src_root_dir:
        #     exc_src_dirs = [os.path.join('!' + src_root_dir, k[1:]) for k in exc_src_dirs]

        # src_dirs = inc_src_dirs + exc_src_dirs
    else:
        src_dirs = [src_dir, ]
        if multi_mode:
            root_dir = os.path.dirname(src_dir)
            src_dir_name = os.path.basename(src_dir)
            src_dirs += [os.path.join(root_dir, k) for k in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, k)) and k != src_dir_name]

    # Process optional counts
    _numerators = []
    _denominators = []
    _src_dirs = []
    _samples = []
    for _id, src_dir in enumerate(src_dirs):
        _numerator = _denominator = 1
        _sample = -1
        if '**' in src_dir:
            _src_dir, _sample = src_dir.split('**')
            src_dir = _src_dir
            _sample = int(_sample)

        if '*' in src_dir:
            _src_dir, _count = src_dir.split('*')
            src_dir = _src_dir
            _numerator = int(_count)
        elif '//' in src_dir:
            _src_dir, _count = src_dir.split('//')
            _denominator = int(_count)
            src_dir = _src_dir
        # if _count > 1:
        #     _src_dirs += [src_dir, ] * _count
        # else:
        _numerators.append(_numerator)
        _denominators.append(_denominator)
        _samples.append(_sample)

        _print(f'{src_dir} : {_numerator} / {_denominator}, {_sample}')

        _src_dirs.append(src_dir)

    lcm = np.lcm.reduce(_denominators)
    _counts = [int(_numerator * lcm / _denominator) for _numerator, _denominator in
               zip(_numerators, _denominators)]
    src_dirs = _src_dirs

    if multi_mode or video_mode:
        n_src = len(src_dirs)
    else:
        n_src = 1

    if video_mode:
        video_files_list = []
        excluded_video_files = []
        n_unique_videos = 0
        for _id, src_dir in enumerate(src_dirs):
            if src_dir[0] == '!':
                src_dir = src_dir[1:]
                excluded = 1
            else:
                excluded = 0

            src_dir = os.path.abspath(src_dir)
            # src_dir = src_dir.replace(os.sep, '/')

            _sample = _samples[_id]
            if _sample <= 0:
                _video_mode = video_mode
            else:
                _video_mode = _sample

            if _video_mode == 2:
                # _print(f'Looking for image sequences in {src_dir}')
                video_file_gen = [[os.path.join(dirpath, d) for d in dirnames if
                                   any([os.path.splitext(f.lower())[1] in img_exts
                                        for f in os.listdir(os.path.join(dirpath, d))])]
                                  for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)]
                _video_files_list = [item for sublist in video_file_gen for item in sublist]

                if not _video_files_list:
                    video_file_gen = [[os.path.join(dirpath, d) for d in dirnames if
                                       any([os.path.splitext(f.lower())[1] in img_exts
                                            for f in os.listdir(os.path.join(dirpath, d))])]
                                      for (dirpath, dirnames, filenames) in os.walk(os.path.dirname(src_dir),
                                                                                    followlinks=True)]
                    _video_files_list = [item for sublist in video_file_gen for item in sublist]

                elif any([os.path.splitext(f.lower())[1] in img_exts
                          for f in os.listdir(src_dir)]):
                    _video_files_list.append(src_dir)
            else:
                # _print(f'Looking for videos in {src_dir}')
                # recursive = 0
                if recursive:
                    # _print(f'Searching recursively')
                    video_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                                       os.path.splitext(f.lower())[1] in vid_exts]
                                      for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)]
                    _video_files_list = [item for sublist in video_file_gen for item in sublist]
                else:
                    _video_files_list = [os.path.join(src_dir, k) for k in os.listdir(src_dir) if
                                         os.path.splitext(k.lower())[1] in vid_exts]
                    _all_files_list = [os.path.join(src_dir, k) for k in os.listdir(src_dir)]

                    _all_files_ext = [os.path.splitext(k.lower())[1] for k in _all_files_list]
                    _all_files_ext_status = [k in vid_exts for k in _all_files_ext]

                    _video_files_list = [os.path.join(src_dir, k) for k in _all_files_list if
                                         os.path.splitext(k.lower())[1] in vid_exts]

            n_videos = len(_video_files_list)

            if excluded:
                _print(f'Excluding {n_videos} videos from: {src_dir}')
                excluded_video_files += _video_files_list
            else:
                if excluded_video_files:
                    _video_files_list = [k for k in _video_files_list if k not in excluded_video_files]
                    n_videos = len(_video_files_list)

                n_unique_videos += n_videos
                if n_videos:
                    # print(f'Found {n_videos} videos in {src_dir}')
                    _print(f'Adding {n_videos} videos from: {src_dir} '
                           f'with multiplicity {_counts[_id]} '
                           f'for total: {int(n_videos * _counts[_id])}')
                    video_files_list += _video_files_list * _counts[_id]
                else:
                    _print(f'Found no videos in {src_dir}')
                    # _print(''
                    #        '_video_files_list:\n{}\n'
                    #        'all_files_list:\n{}\n'
                    #        'all_files_ext:\n{}\n'
                    #        '_all_files_ext_status:\n{}\n'
                    #        'vid_exts:\n{}\n'.format(
                    #     pformat(_video_files_list),
                    #     pformat(_all_files_list),
                    #     pformat(_all_files_ext),
                    #     pformat(_all_files_ext_status),
                    #     vid_exts,
                    # )
                    # )

        if not video_files_list:
            raise IOError('No videos found in any source folder')

        try:
            video_files_list.sort(key=sortKey)
        except:
            video_files_list.sort()

        # print(f'video_files_list:\n {pformat(video_files_list)}')

        if random_mode:
            video_files_list = list(np.random.permutation(video_files_list))

        try:
            vid_id = video_files_list.index(src_path)
        except ValueError:
            vid_id = 0
            src_path = video_files_list[0]

        n_videos = len(video_files_list)
        if n_videos > 1:
            _print(f'Found a total of {n_videos} videos (unique: {n_unique_videos})')
        else:
            _print(f'Found no videos')

    if not video_mode or images_as_video:

        print(f'src_dirs:\n {pformat(src_dirs)}')

        excluded_src_files = []
        for _id, src_dir in enumerate(src_dirs):

            if _samples[_id] < 0:
                _samples[_id] = 1

            if src_dir[0] == '!':
                src_dir = src_dir[1:]
                excluded = 1
            else:
                excluded = 0

            # print(f'{src_dir} : count: { _counts[_id]}')

            if recursive:
                src_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                                 os.path.splitext(f.lower())[1] in img_exts]
                                for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)]
                _src_files = [item for sublist in src_file_gen for item in sublist]

                # _src_file_list = list(src_file_gen)
                # src_file_list = []
                # for x in _src_file_list:
                #     src_file_list += x
            else:
                _src_files = [os.path.join(src_dir, k) for k in os.listdir(src_dir) if
                              os.path.splitext(k.lower())[1] in img_exts]

            _src_files = [os.path.abspath(k) for k in _src_files]

            _n_src_files = len(_src_files)
            if excluded:
                _print(f'Excluding {_n_src_files} images from: {src_dir}')
                excluded_src_files += _src_files
            else:
                if excluded_src_files:
                    _src_files = [k for k in _src_files if k not in excluded_src_files]
                    _n_src_files = len(_src_files)
                _print(f'Adding {_n_src_files} images from: {src_dir} '
                       f'with sample: {_samples[_id]} and multiplicity {_counts[_id]} '
                       f'for total: {int(_n_src_files * _counts[_id] / _samples[_id])}')
                src_files[_id] = _src_files

            # src_file_list = [list(x) for x in src_file_list]
            # src_file_list = [x for x in src_file_list]

            # print('src_file_list: ', src_file_list)

            # for (dirpath, dirnames, filenames) in os.walk(src_path):
            #     print()
            #     print('dirpath', dirpath)
            #     print('filenames', filenames)
            #     print('dirnames', dirnames)
            #     print()
        for _id in src_files:
            # if excluded_src_files:
            #     src_files[_id] = [k for k in src_files[_id] if k not in excluded_src_files]

            if _samples[_id] > 1:
                src_files[_id] = src_files[_id][::_samples[_id]]

            total_frames[_id] = len(src_files[_id])
            try:
                # nums = int(os.path.splitext(img_fname)[0].split('_')[-1])
                src_files[_id].sort(key=img_sortKey)
            except:
                src_files[_id].sort()

            if not multi_mode and _id > 0:
                total_frames[0] += total_frames[_id] * _counts[_id]
                src_files[0] += src_files[_id] * _counts[_id]

            if random_mode:
                src_files_rand[_id] = list(np.random.permutation(src_files[_id]))
            # print('src_file_list: {}'.format(src_file_list))
            # print('img_fname: {}'.format(img_fname))
            # print('img_id: {}'.format(img_id))
            _total_frames = total_frames[_id]

            img_id[_id] = 0

            if _total_frames <= 0:
                raise IOError('No input frames found for _id: {}'.format(_id))
            # print('Found {} frames'.format(_total_frames))

        if not multi_mode and random_mode:
            src_files_rand[0] = list(np.random.permutation(src_files[0]))
            _print('total_frames: {}'.format(total_frames[0]))

        if img_fname is None:
            img_fname = src_files[0][img_id[0]]

        img_id[0] = src_files[0].index(img_fname)

        if video_mode:
            video_files_list += src_files[0]
            n_videos = len(video_files_list)

    if video_mode:
        loadVideo(0)
        transition_interval = int(1000.0 / fps)
        total_frames = {
            0: total_frames[0]
        }
        _total_frames = total_frames[0]

    _print('transition_interval: {}'.format(transition_interval))

    if not multi_mode:
        _print(f'total_frames: {total_frames[0]}')

    if not multi_mode:
        img_id = {
            0: img_id[0]
        }
    src_img_ar, start_row, end_row, start_col, end_col, dst_height, dst_width = [None] * 7
    target_height, target_width, min_height, start_col, end_col, height_ratio = [None] * 6
    dst_img = None

    prev_active_handle = None
    prev_active_win_name = None
    active_monitor_id = None

    script_filename = inspect.getframeinfo(inspect.currentframe()).filename
    script_path = os.path.dirname(os.path.abspath(script_filename))

    log_dir = os.path.join(script_path, 'log')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'vwm_log.txt')
    _print('Saving log to {}'.format(log_file))

    if not wallpaper_dir:
        wallpaper_dir = os.path.join(log_dir, 'vwm')
    if not os.path.isdir(wallpaper_dir):
        os.makedirs(wallpaper_dir)
    _print('Saving wallpapers to {}'.format(wallpaper_dir))

    auto_progress_type = 0

    if not video_mode and check_images:
        _print('Checking images...')
        for _set_id in src_files:
            _n_images = len(src_files[_set_id])
            for _img_id, img_path in enumerate(src_files[_set_id]):
                code, output, error = checkImage(img_path)
                if str(code) != "0" or str(error, "utf-8") != "":
                    _print("\n{} :: ERROR: {}\n".format(img_path, error))
                sys.stdout.write('\r Set {}: Done {}/{}'.format(
                    _set_id + 1, _img_id + 1, _n_images))

    def createWindow(_win_name):
        nonlocal mode, move_to_right

        try:
            cv2.destroyWindow(_win_name)
        except:
            pass

        if mode == 0:
            if fullscreen:
                cv2.namedWindow(_win_name, cv2.WND_PROP_FULLSCREEN)

                # if duplicate_window:
                #     cv2.namedWindow(_win_name2, cv2.WND_PROP_FULLSCREEN)

                cv2.setWindowProperty(_win_name, cv2.WND_PROP_FULLSCREEN, 1)
            else:
                cv2.namedWindow(_win_name, cv_windowed_mode_flags)

                # if duplicate_window:
                #     cv2.namedWindow(_win_name2, cv_windowed_mode_flags)

                # hideBorder()
                if win_utils_available:
                    winUtils.hideBorder2(_win_name, on_top)
                    # winUtils.hideBorder(monitors[curr_monitor][0], monitors[curr_monitor][1],
                    #                     width, height, _win_name)

            if frg_win_titles:
                cv2.moveWindow(_win_name, frg_positions[frg_win_id][0], frg_positions[frg_win_id][1])
            else:
                cv2.moveWindow(_win_name, win_offset_x + monitors[monitor_id][0],
                               win_offset_y + monitors[monitor_id][1])
        else:
            cv2.namedWindow(_win_name, cv_windowed_mode_flags)

            # if duplicate_window:
            #     cv2.namedWindow(_win_name2, cv_windowed_mode_flags)

            #     winUtils.hideBorder(monitors[2][0], monitors[2][1], width, height, _win_name)
            # else:
            # hideBorder()
            if win_utils_available:
                winUtils.hideBorder2(_win_name, on_top)
                # winUtils.loseFocus(_win_name)
            if frg_win_titles:
                cv2.moveWindow(_win_name, frg_positions[frg_win_id][0], frg_positions[frg_win_id][1])
            else:
                if widescreen_mode:
                    cv2.moveWindow(_win_name, win_offset_x + widescreen_monitor[0],
                                   win_offset_y + widescreen_monitor[1])
                else:
                    if move_to_right:
                        cv2.moveWindow(_win_name, win_offset_x + monitors[4][0], win_offset_y + monitors[4][1])
                    else:
                        cv2.moveWindow(_win_name, win_offset_x + monitors[2][0], win_offset_y + monitors[2][1])

        cv2.setMouseCallback(_win_name, mouseHandler)

        # if hotkeys_available:
        #     HOTKEYS = {
        #         1: (win32con.VK_F3, win32con.MOD_WIN),
        #         2: (win32con.VK_F4, win32con.MOD_WIN)
        #     }
        #     for _id, (vk, modifiers) in HOTKEYS.items():
        #         print("Registering id", id, "for key", vk)
        #         if not user32.RegisterHotKey(None, _id, modifiers, vk):
        #             print("Unable to register id", _id)

    def changeMode():
        nonlocal mode, height, width, aspect_ratio, widescreen_mode
        if widescreen_mode:
            width = 5760
            height = 2160
        else:
            width = 1920
            mode = 1 - mode
            if mode == 0:
                height = 1080
            else:
                height = 2160

        _print('changeMode :: height: ', height)
        _print('changeMode :: width: ', width)

        aspect_ratio = float(width) / float(height)
        createWindow(win_name)
        if duplicate_window:
            for _win_name2 in dup_win_names:
                createWindow(_win_name2)

        loadImage()

    def setGridSize():
        nonlocal grid_size, n_images, predef_grid_sizes
        try:
            n_rows, n_cols = predef_grid_sizes[n_images]
        except KeyError:
            if n_images % 3 == 0:
                n_cols = 3
                n_rows = int(n_images / 3)
            elif n_images % 2 == 0 and n_images > 2:
                n_rows = 2
                n_cols = int(n_images / 2)
            else:
                n_rows = 1
                n_cols = n_images
                # n_cols = n_rows = int(np.ceil(np.sqrt(n_images)))
        grid_size = (n_rows, n_cols)

    def loadImage(_type=0, set_grid_size=0, decrement_id=0):
        nonlocal src_img_ar, start_row, end_row, start_col, end_col, dst_height, dst_width, n_switches, img_id, direction
        nonlocal target_height, target_width, min_height, start_col, end_col, height_ratio, img_fname, start_time, video_files_list
        nonlocal src_start_row, src_start_col, src_end_row, src_end_col, aspect_ratio, src_path, vid_id, \
            src_images, img_fnames, stack_idx, stack_locations, src_img, wp_id, src_files_rand, top_border, bottom_border
        nonlocal img_sequences

        if decrement_id:
            if video_mode:
                for _id in img_id:
                    img_id[_id] -= 1
            else:
                for _id in img_id:
                    img_id[_id] -= 1
        if set_grid_size:
            setGridSize()

        if _type != 0 and not keep_borders:
            top_border = bottom_border = 0
        aspect_ratio = float(width) / float(height)

        if _type != 0 or not src_images or video_mode:
            if video_mode or multi_mode:
                for _id in img_id:
                    if _type == 0:
                        img_id[_id] -= 1
                    elif _type == -1:
                        img_id[_id] -= 2
            else:
                if _type == 0:
                    img_id[0] -= n_images
                elif _type == -1:
                    img_id[0] -= 2 * n_images
            src_images = []
            img_fnames = {}
            for _load_id in range(n_images):
                if video_mode:
                    if _load_id not in total_frames:
                        vid_id = vid_id + 1
                        if vid_id >= n_videos:
                            if random_mode:
                                _print('Resetting randomisation')
                                video_files_list = list(np.random.permutation(video_files_list))
                            vid_id = 0
                        src_path = video_files_list[vid_id]
                        loadVideo(_load_id)
                    _total_frames = total_frames[_load_id]
                    _img_id = img_id[_load_id]
                    src_id = 0
                else:
                    src_id = _load_id % n_src
                    _total_frames = total_frames[src_id]
                    _img_id = img_id[src_id]
                _img_id += 1

                # if _type == 1:
                #     # if random_mode:
                #     #     img_id += random.randint(1, total_frames)
                #     # else:
                #
                # elif _type == -1:
                #     # if random_mode:
                #     #     img_id -= random.randint(1, total_frames)
                #     # else:
                #     img_id -= 1

                if _img_id >= _total_frames:
                    if video_mode and auto_progress_video:
                        vid_id = (vid_id + 1) % n_videos
                        src_path = video_files_list[vid_id]
                        loadVideo(_load_id)
                        _img_id = 0
                    else:
                        if video_mode and reverse_video and (video_mode == 2 or not lazy_video_load):
                            src_files[_load_id] = list(reversed(src_files[_load_id]))
                        _img_id -= _total_frames
                        if not video_mode and auto_progress and random_mode:
                            _print('Resetting randomisation')
                            src_files_rand[src_id] = list(np.random.permutation(src_files[src_id]))
                elif _img_id < 0:
                    _img_id += _total_frames

                if video_mode:
                    if isinstance(src_files[_load_id], VideoCapture):
                        # start_t = time.time()
                        ret, src_img = src_files[_load_id].read()
                        # end_t = time.time()
                        # print(f'fps: {1.0 / (end_t - start_t)}')
                        if not ret:
                            src_files[_load_id].set(cv2.CAP_PROP_POS_FRAMES, 0)
                            # src_files[_load_id].release()
                            # if auto_progress_video:
                            #     vid_id = (vid_id + 1) % n_videos
                            # src_path = video_files_list[vid_id]
                            # loadVideo(_load_id)
                            _img_id = 0
                            ret, src_img = src_files[_load_id].read()
                    else:
                        img_fname = src_files[_load_id][_img_id]
                        if isinstance(img_fname, str):
                            if parallel_read:
                                while True:
                                    try:
                                        src_img = img_sequences[_load_id][img_fname]
                                    except KeyError:
                                        continue
                                    else:
                                        break
                            else:
                                if not os.path.isfile(img_fname):
                                    # _exit_neatly()
                                    _print('Video frame does not exist: {}'.format(img_fname))
                                    return
                                try:
                                    src_img = img_sequences[_load_id][img_fname]
                                except KeyError:
                                    src_img = cv2.imread(img_fname)
                                    img_sequences[_load_id][img_fname] = src_img
                        else:
                            src_img = np.copy(img_fname)
                    # if trim_images:
                    #     src_img = np.asarray(trim(Image.fromarray(src_img)))
                    if rotate_images:
                        src_img = np.rot90(src_img, rotate_images)
                else:
                    if random_mode:
                        img_fname = src_files_rand[src_id][_img_id]
                    else:
                        img_fname = src_files[src_id][_img_id]
                    # src_img_fname = os.path.join(src_dir, img_fname)

                    # print('img_id: {}'.format(img_id))
                    # print('img_fname: {}'.format(img_fname))
                    src_img_fname = img_fname
                    if not os.path.isfile(src_img_fname):
                        # _exit_neatly()
                        _print('Source image does not exist: {}'.format(src_img_fname))
                        return
                    src_img = cv2.imread(src_img_fname)
                    if trim_images:
                        src_img = np.asarray(trim(Image.fromarray(src_img)))
                        # src_img = wandImage(src_img).trim(color=None, fuzz=0) ()

                    if rotate_images:
                        src_img = np.rot90(src_img, rotate_images)
                    img_fnames[_load_id] = img_fname

                src_images.append(src_img)
                if video_mode:
                    img_id[_load_id] = _img_id
                else:
                    img_id[src_id] = _img_id

        if n_images == 1:
            src_img = src_images[0]
            if top_border > 0:
                src_img = addBorder(src_img, top_border, 'top')
            if bottom_border > 0:
                src_img = addBorder(src_img, bottom_border, 'bottom')
        else:
            src_img, stack_idx, stack_locations = stackImages(src_images, grid_size, borderless=borderless,
                                                              return_idx=1, preserve_order=preserve_order)
            # print('stack_locations: {}'.format(stack_locations))

        if alpha < 1:
            src_img = (alpha * src_img).astype(np.uint8)
        if set_wallpaper:
            if not wallpaper_mode:
                wp_id = (wp_id + 1) % n_wallpapers
            wp_fname = os.path.join(wallpaper_dir, 'wallpaper_{}.jpg'.format(wp_id))

            if set_wallpaper == 3:
                out_wp_width, out_wp_height = width, height
                src_img_desktop = src_img
                wp_start_row = wp_start_col = 0
                wp_width, wp_height = width, height
                border_type = 'bottom'
                wp_border = 30
            else:
                screensize = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
                out_wp_width, out_wp_height = screensize
                wp_width = 1920

                if set_wallpaper == 1:
                    if screensize[0] == 1920 and screensize[1] == 1080:
                        wp_border = 30
                    else:
                        wp_border = 0
                    wp_height = 1080
                    wp_start_row = screensize[1] - 1080
                    wp_start_col = 0
                else:
                    wp_border = 30
                    wp_height = screensize[1]
                    wp_start_row = 0
                    if screensize[0] >= 3840:
                        wp_start_col = 1920
                    else:
                        wp_start_col = 0
                if n_images == 1 or grid_size[0] % 2 == 1:
                    border_type = 'bottom'
                else:
                    border_type = 'top_and_bottom'

            if wp_border:
                wp_height_ratio = float(src_img.shape[0]) / float(wp_height)
                src_border = int(wp_border * wp_height_ratio)
                src_img = addBorder(src_img, src_border, border_type)

            src_img_desktop = resizeAR(src_img, wp_width, wp_height, placement_type=reversed_pos)
            # src_img = addBorder(src_img, bottom_border, 1)

            wp_end_col = wp_start_col + src_img_desktop.shape[1]
            wp_end_row = wp_start_row + src_img_desktop.shape[0]
            src_img_desktop_full = np.zeros((out_wp_height, out_wp_width, 3), dtype=np.uint8)
            src_img_desktop_full[wp_start_row:wp_end_row, wp_start_col:wp_end_col, :] = src_img_desktop
            cv2.imwrite(wp_fname, src_img_desktop_full)
            win_wallpaper_func(SPI_SETDESKWALLPAPER, 0, wp_fname, 0)

        src_height, src_width, n_channels = src_img.shape

        # if mode == 1 and src_height < src_width:
        #     src_img = np.rot90(src_img)
        #     src_height, src_width, n_channels = src_img.shape

        src_aspect_ratio = float(src_width) / float(src_height)

        if src_aspect_ratio == aspect_ratio:
            dst_width = src_width
            dst_height = src_height
            start_row = start_col = 0
        elif src_aspect_ratio > aspect_ratio:
            dst_width = src_width
            dst_height = int(src_width / aspect_ratio)
            if reversed_pos == 0:
                start_row = 0
            elif reversed_pos == 1:
                start_row = int((dst_height - src_height) / 2.0)
            elif reversed_pos == 2:
                start_row = int(dst_height - src_height)
            start_col = 0
        else:
            dst_height = src_height
            dst_width = int(src_height * aspect_ratio)
            if reversed_pos == 0:
                start_col = 0
            elif reversed_pos == 1:
                start_col = int((dst_width - src_width) / 2.0)
            elif reversed_pos == 2:
                start_col = int(dst_width - src_width)
            start_row = 0

        # if mode == 0:
        # else:
        #     if src_aspect_ratio == aspect_ratio:
        #         dst_width = width
        #         dst_height = height
        #     elif src_aspect_ratio > aspect_ratio:
        #         # too tall
        #         dst_height = int(height)
        #         dst_width = int(height * src_aspect_ratio)
        #     else:
        #         # too wide
        #         dst_width = int(width)
        #         dst_height = int(width / aspect_ratio)

        # src_img = np.zeros((height, width, n_channels), dtype=np.uint8)

        src_start_row = start_row
        src_start_col = start_col
        src_end_row = start_row + src_height
        src_end_col = start_col + src_width

        src_img_ar = np.zeros((dst_height, dst_width, n_channels), dtype=np.uint8)
        src_img_ar[int(src_start_row):int(src_end_row), int(src_start_col):int(src_end_col), :] = src_img

        target_width = dst_width
        target_height = dst_height

        start_row = start_col = 0
        end_row = dst_height
        end_col = dst_width

        min_height = dst_height * min_height_ratio

        height_ratio = float(dst_height) / float(height)

        n_switches = 0
        direction = -1

        # if show_window:
        #     keyboard.send('y')

        start_time = time.time()

        # print('height: ', height)
        # print('dst_height: ', dst_height)
        # print('dst_width: ', dst_width)

    # def motionStep(_direction):
    #     global target_height, direction, end_row, start_col, end_col
    #
    #     target_height = target_height + _direction * speed
    #
    #     if target_height < min_height:
    #         target_height = min_height
    #         _direction = 1
    #
    #     if target_height > dst_height:
    #         target_height = dst_height
    #         _direction = -1
    #
    #     target_width = target_height * aspect_ratio
    #
    #     # print('speed: ', speed)
    #     # print('min_height: ', min_height)
    #     # print('target_height: ', target_height)
    #     # print('target_width: ', target_width)
    #
    #     end_row = start_row + target_height
    #
    #     col_diff = (dst_width - target_width) / 2.0
    #     start_col = col_diff
    #     end_col = dst_width - col_diff
    #
    #     return _direction

    def increaseSpeed():
        nonlocal speed
        speed += 0.05
        _print('speed: ', speed)

    def decreaseSpeed():
        nonlocal speed
        speed -= 0.05
        if speed < 0:
            speed = 0
        _print('speed: ', speed)

    def setOffsetDiff(dx, dy):
        nonlocal row_offset, col_offset

        curr_width = end_col - start_col + 1
        col_offset += dx * float(curr_width) / float(width)
        if end_col + col_offset > dst_width:
            col_offset -= end_col + col_offset - dst_width
        elif col_offset + start_col < 0:
            col_offset = -start_col

        curr_height = end_row - start_row + 1
        row_offset += dy * float(curr_height) / float(height)

        if end_row + row_offset > dst_height:
            row_offset -= end_row + row_offset - dst_height
        elif row_offset + start_row < 0:
            row_offset = -start_row

    def setOffset(x, y):
        nonlocal row_offset, col_offset

        curr_width = end_col - start_col + 1
        col_offset = col_offset + (x * float(curr_width) / float(width))
        # print('start_offset: {}'.format(start_offset))

        if end_col + col_offset > dst_width:
            col_offset -= end_col + col_offset - dst_width

        col_offset -= dst_width / 2.0
        if col_offset + start_col < 0:
            col_offset = - start_col
        # print('start_row: {}'.format(start_row))
        # print('height_ratio: {}'.format(height_ratio))
        # print('dst_height: {}'.format(dst_height))
        # print('start_offset: {}'.format(start_offset))

        curr_height = end_row - start_row + 1
        row_offset = row_offset + (y * float(curr_height) / float(height))
        # print('start_offset: {}'.format(start_offset))

        if end_row + row_offset > dst_height:
            row_offset -= end_row + row_offset - dst_height

        # print('start_row: {}'.format(start_row))
        # print('height_ratio: {}'.format(height_ratio))
        # print('dst_height: {}'.format(dst_height))
        # print('start_offset: {}'.format(start_offset))

    def updateZoom(_speed=None, _direction=None):
        nonlocal target_height, direction, start_col, start_row, end_row, end_col, n_switches

        if _speed is None:
            _speed = speed if mode == 0 else 2 * speed

        if _direction is None:
            _direction = direction

        target_height = target_height + _direction * _speed * height_ratio

        if target_height < min_height:
            target_height = min_height
            direction = 1

        if target_height > dst_height:
            target_height = dst_height
            n_switches += 1
            if auto_progress and n_switches >= max_switches:
                loadImage(1)
            else:
                direction = -1

        target_width = target_height * aspect_ratio

        # print('speed: ', speed)
        # print('min_height: ', min_height)
        # print('target_height: ', target_height)
        # print('target_width: ', target_width)

        end_row = start_row + target_height

        col_diff = (dst_width - target_width) / 2.0
        start_col = col_diff
        end_col = dst_width - col_diff

    def minimizeWindow():
        try:
            win_handle = ctypes.windll.user32.FindWindowW(None, win_name)
            ctypes.windll.user32.ShowWindow(win_handle, 6)
        except:
            _print('Window minimization unavailable')

    def maximizeWindow():
        try:
            win_handle = ctypes.windll.user32.FindWindowW(None, win_name)
            ctypes.windll.user32.ShowWindow(win_handle, 1)
        except:
            _print('Window minimization unavailable')

    def getClickedImage(x, y, get_idx=0):
        if video_mode:
            return 'Frame {}'.format(img_id)

        if n_images == 1:
            if get_idx:
                return img_fname, 0
            return img_fname
            # return img_fname if not get_idx else img_fname, 0
        resize_ratio = float(dst_img.shape[0]) / float(src_img.shape[0])
        x_scaled, y_scaled = x / resize_ratio, y / resize_ratio
        for i in range(n_images):
            _start_row, _start_col, _end_row, _end_col = stack_locations[i]
            if x_scaled >= _start_col and x_scaled < _end_col and y_scaled >= _start_row and y_scaled < _end_row:
                __idx = stack_idx[i]
                fname = os.path.abspath(img_fnames[__idx])
                _print('Clicked on image {} with id {}:\n {}'.format(i + 1, __idx, fname))
                if get_idx:
                    return fname, __idx
                return fname
                # return fname if not get_idx else fname, __idx

        _print('Image for the clicked point {}, {} not found'.format(x, y))

        if get_idx:
            return None, None
        return None
        # return None if not get_idx else None, None

    def sortImage(_img_name, sort_type):
        if _img_name is None:
            return
        if _img_name in images_to_sort_inv:
            prev_key = images_to_sort_inv[_img_name]
            if prev_key != sort_type:
                _print('Removing previous sorting of {} into {}'.format(_img_name, prev_key))
                images_to_sort[prev_key].remove(_img_name)
                del images_to_sort_inv[_img_name]
        _print('Sorting {} into category {}'.format(_img_name, sort_type))
        try:
            images_to_sort[sort_type].append(_img_name)
        except KeyError:
            images_to_sort[sort_type] = [_img_name, ]
        images_to_sort_inv[_img_name] = sort_type
        if n_images == 1:
            loadImage(1)

    def mouseHandler(event, x, y, flags=None, param=None):
        nonlocal img_id, row_offset, col_offset, lc_start_t, rc_start_t, end_exec, fullscreen, \
            direction, target_height, prev_pos, prev_win_pos, speed, old_speed, min_height, min_height_ratio, n_images, src_images
        nonlocal win_offset_x, win_offset_y, width, height, top_border, bottom_border, images_to_sort, \
            images_to_sort_inv, auto_progress, src_files, rotate_images, src_path, vid_id, auto_progress_video
        reset_prev_pos = reset_prev_win_pos = True

        # print('event: {}'.format(event))

        try:
            if event == cv2.EVENT_MBUTTONDBLCLK:
                end_exec = 1
            elif event == cv2.EVENT_RBUTTONDBLCLK:
                pass
                # start_row = y
                # fullscreen = 1 - fullscreen
                # createWindow()
                # if fullscreen:
                #     print('fullscreen mode enabled')
                # else:
                #     print('fullscreen mode disabled')
                # loadImage(-1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                # if  rc_start_t is None:
                #     rc_start_t = time.time()
                # else:
                #     rc_end_t = time.time()
                #     click_interval = rc_end_t - rc_start_t
                #     if click_interval < double_click_interval:
                #         end_exec = 1
                #     rc_start_t = None
                # print('flags: {}'.format(flags))
                # print('flags: {0:b}'.format(flags))

                flags_to_sort_type = {
                    34: ('2', 'alt'),
                    42: ('4', 'alt + ctrl'),
                    58: ('6', 'alt + ctrl + shift'),
                }
                if flags == 2:
                    if video_mode and auto_progress:
                        vid_id = (vid_id + 1) % n_videos
                        src_path = video_files_list[vid_id]
                        loadVideo(0)
                        loadImage()
                    else:
                        _print('next image')
                        loadImage(1)
                elif flags == 10 or flags == 11:
                    direction = -direction
                elif flags == 18:
                    row_offset = col_offset = 0
                elif flags in flags_to_sort_type.keys():
                    _img_fname = getClickedImage(x, y)
                    sort_type = flags_to_sort_type[flags][0]
                    sortImage(_img_fname, sort_type)
            elif event == cv2.EVENT_RBUTTONUP:
                pass
            elif event == cv2.EVENT_MBUTTONDOWN:
                flags_str = '{0:b}'.format(flags)
                # print('EVENT_MBUTTONDOWN flags: {:s}'.format(flags_str))
                if flags_str == '100':
                    if video_mode:
                        auto_progress = 1 - auto_progress
                    loadImage()
                elif flags_str == '1100':
                    # ctrl
                    target_height = min_height
                elif flags_str == '10100':
                    # shift
                    if video_mode:
                        _print('Reversing video')
                        for _id in img_id:
                            img_id[_id] = total_frames[_id] - img_id[_id] - 1
                            src_files[_id] = list(reversed(src_files[_id]))
                elif flags_str == '100100':
                    # alt
                    rotate_images += 1
                    if rotate_images > 3:
                        rotate_images = 0
                    _print('Rotating images by {} degrees'.format(rotate_images * 90))
                    src_images = []
                    loadImage()
                elif flags_str == '11100':
                    # ctrl + shift
                    auto_progress_video = 1 - auto_progress_video
                    if auto_progress_video:
                        _print('Video auto progression enabled')
                    else:
                        _print('Video auto progression disabled')

            elif event == cv2.EVENT_MOUSEMOVE:
                # print('EVENT_MOUSEMOVE flags: {}'.format(flags))
                if flags == 33:
                    reset_prev_win_pos = False
                    if prev_win_pos is None:
                        prev_win_pos = [x, y]
                        return
                    win_offset_x += x - prev_win_pos[0]
                    win_offset_y += y - prev_win_pos[1]

                    # print('win_offset_x: {}'.format(win_offset_x))
                    # print('win_offset_y: {}'.format(win_offset_y))

                    prev_win_pos = [x, y]
                elif flags == 9 or flags == 17 or flags == 11:
                    if flags == 11:
                        direction = -direction
                        # if speed == 0:
                        #     speed = old_speed
                        # else:
                        #     old_speed = speed
                        #     speed = 0
                    elif flags == 9:
                        # target_height = min_height
                        if target_height == dst_height:
                            target_height = min_height
                    reset_prev_pos = False
                    if prev_pos is None:
                        prev_pos = [x, y]
                        return
                    pos_diff_x = x - prev_pos[0]
                    pos_diff_y = y - prev_pos[1]
                    prev_pos = [x, y]
                    setOffsetDiff(-pos_diff_x, -pos_diff_y)
            elif event == cv2.EVENT_MOUSEWHEEL:
                keys_to_flags = {
                    'ctrl': (7864328, -7864312),
                    'alt': (7864352, -7864288),
                    'shift': (7864336, -7864304),
                    'ctrl+alt': (7864360, -7864280),
                    'ctrl+shift': (7864344, -7864296),
                    'alt+shift': (7864368, -7864272),
                    'ctrl+alt+shift': (7864376, -7864264),
                }

                # flags_str = '{0:b}'.format(flags)

                # print('EVENT_MOUSEWHEEL flags: {}'.format(flags))
                # print('EVENT_MOUSEWHEEL flags_str: {:s}'.format(flags_str))

                # _delta = cv2.getMouseWheelDelta(flags)
                if flags > 0:
                    if flags == keys_to_flags['alt'][0]:
                        height += 5
                        loadImage()
                        pass
                    elif flags == keys_to_flags['alt+shift'][0]:
                        width += 5
                        loadImage()
                    elif flags == keys_to_flags['ctrl'][0]:
                        min_height_ratio -= 0.01
                        if min_height_ratio < 0.01:
                            min_height_ratio = 0.01
                        target_height = min_height = min_height_ratio * dst_height
                    elif flags == keys_to_flags['shift'][0]:
                        row_offset += dst_height * 0.01
                        if end_row + row_offset > dst_height:
                            row_offset -= end_row + row_offset - dst_height
                    elif flags == keys_to_flags['ctrl+shift'][0]:
                        col_offset += dst_width * 0.01
                        if end_col + col_offset > dst_width:
                            col_offset -= end_col + col_offset - dst_width
                    else:
                        if n_images == 1:
                            if is_paused:
                                top_border -= 5
                                if top_border < 0:
                                    top_border = 0
                                bottom_border += 5
                                loadImage()
                            else:
                                increaseSpeed()
                                # motionStep(1)
                        else:
                            loadImage(-1)
                else:
                    if flags == keys_to_flags['alt'][1]:
                        height -= 5
                        if height < 10:
                            height = 10
                        loadImage()
                    elif flags == keys_to_flags['alt+shift'][1]:
                        width -= 5
                        if width < 10:
                            width = 10
                        loadImage()
                    elif flags == keys_to_flags['ctrl'][1]:
                        min_height_ratio += 0.01
                        if min_height_ratio > 1:
                            min_height_ratio = 1
                        target_height = min_height = min_height_ratio * dst_height
                    elif flags == keys_to_flags['shift'][1]:
                        row_offset -= dst_height * 0.01
                        if row_offset + start_row < 0:
                            row_offset = -start_row
                    elif flags == keys_to_flags['ctrl+shift'][1]:
                        col_offset -= dst_width * 0.01
                        if col_offset + start_col < 0:
                            col_offset = -start_col
                    else:
                        if n_images == 1:
                            if is_paused:
                                bottom_border -= 5
                                if bottom_border < 0:
                                    bottom_border = 0
                                top_border += 5
                                loadImage()
                            else:
                                decreaseSpeed()
                                # motionStep(1)
                        else:
                            loadImage(1)
            if reset_prev_pos and reset_prev_win_pos:
                prev_pos = prev_win_pos = None
                if event == cv2.EVENT_LBUTTONDOWN:
                    # if lc_start_t is None:
                    #     lc_start_t = time.time()
                    # else:
                    #     lc_end_t = time.time()
                    #     click_interval = lc_end_t - lc_start_t
                    #     print('click_interval: ', click_interval)
                    #     if click_interval < double_click_interval:
                    #         lc_start_t = None
                    # print('flags: {}'.format(flags))
                    # print('flags_b: {0:b}'.format(flags))

                    flags_to_sort_type = {
                        33: ('1', 'alt'),
                        41: ('3', 'alt + ctrl'),
                        57: ('5', 'alt + ctrl + shift'),
                    }
                    keys_to_flags = {
                        'none': 1,
                        'ctrl': 9,
                        'shift': 17,
                        'ctrl+shift': 25,
                        'alt': 33,
                        'alt+ctrl': 41,
                        'alt+ctrl+shift': 57,
                    }
                    if flags == 1:
                        if video_mode and auto_progress:
                            vid_id -= 1
                            if vid_id < 0:
                                vid_id = n_videos - 1
                            src_path = video_files_list[vid_id]
                            loadVideo(0)
                            loadImage()
                        else:
                            loadImage(-1)
                    elif flags == 9:
                        # ctrl
                        target_height = min_height
                    elif flags in flags_to_sort_type.keys():
                        _img_fname = getClickedImage(x, y)
                        sort_type = flags_to_sort_type[flags][0]
                        sortImage(_img_fname, sort_type)
                    elif flags == 17 or flags == 25:
                        # shift
                        # print('n_images: {}'.format(n_images))
                        if not video_mode:
                            if n_images > 1:
                                # print('here we are')
                                clicked_img_fname, __idx = getClickedImage(x, y, get_idx=1)
                                if clicked_img_fname is not None:
                                    fname = '"' + clicked_img_fname + '"'
                                    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
                                    open(log_file, 'a').write(time_stamp + "\n" + fname + '\n')
                                    if flags == 25:
                                        # ctrl + shift
                                        if not video_mode:
                                            img_id[0] += __idx + 1 - n_images
                                            # print('making img_id: {}'.format(img_id))
                                            n_images = 1
                                            src_images = []
                                            loadImage(0)
                                else:
                                    resize_ratio = float(dst_img.shape[0]) / float(src_img.shape[0])
                                    x_scaled, y_scaled = x / resize_ratio, y / resize_ratio
                                    # click_found = 0
                                    # for i in range(n_images):
                                    #     _start_row, _start_col, _end_row, _end_col = stack_locations[i]
                                    #     if x_scaled >= _start_col and x_scaled < _end_col and y_scaled >= _start_row and y_scaled < _end_row:
                                    #         __idx = stack_idx[i]
                                    #         fname = '"' + os.path.abspath(img_fnames[__idx]) + '"'
                                    #         print('Clicked on image {} with id {}:\n {}'.format(i + 1, __idx, fname))
                                    #         time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
                                    #         open(log_file, 'a').write(time_stamp + "\n" + fname + '\n')
                                    #         click_found = 1
                                    #
                                    #         if flags == 25:
                                    #             # ctrl + shift
                                    #             img_id += __idx + 1 - n_images
                                    #             # print('making img_id: {}'.format(img_id))
                                    #             n_images = 1
                                    #             src_images = []
                                    #             loadImage(0)
                                    #         break
                                    # if not click_found:
                                    _print('x: {}'.format(x))
                                    _print('y: {}'.format(y))
                                    _print('resize_ratio: {}'.format(resize_ratio))
                                    _print('x_scaled: {}'.format(x_scaled))
                                    _print('y_scaled: {}'.format(y_scaled))
                                    _print('stack_locations:\n {}\n'.format(stack_locations))
                        elif n_images == 1:
                            vid_name = video_files_list[vid_id]
                            time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
                            open(log_file, 'a').write(time_stamp + "\n" + vid_name + '\n')
                            try:
                                import pyperclip

                                pyperclip.copy(vid_name)
                                _ = pyperclip.paste()
                            except BaseException as err:
                                _print('Copying to clipboard failed: {}'.format(err))
                            else:
                                _print('Copied to clipboard: {}'.format(vid_name))

                    # elif flags == 9:
                    #     row_offset = col_offset = 0
                    # setOffset(x, y)
                    # elif flags == 17:
                    #     row_offset = col_offset = 0
                elif event == cv2.EVENT_LBUTTONUP:
                    pass

        except AttributeError as e:
            _print('AttributeError: {}'.format(e))
            pass

    if not win_name:
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        win_name = 'VWM_{}_{}'.format(os.path.basename(os.path.abspath(src_path)),
                                      time_stamp)
    dup_win_names = []
    for _i, _ in enumerate(dup_monitor_ids):
        dup_win_names.append('{} {}'.format(win_name, _i))

    # win_names_file = os.path.join(log_dir, 'vwm_win_names.txt')
    # print('Writing win_names to {}'.format(log_file))
    #
    # with open(win_names_file, 'w') as fid:
    #     fid.write(win_name + '\n')
    #     fid.write(dup_win_names + '\n')

    if not wallpaper_mode:
        createWindow(win_name)
        if duplicate_window:
            for _win_name2 in dup_win_names:
                createWindow(_win_name2)

    MAX_TRANSITION_INTERVAL = 1000
    MIN_TRANSITION_INTERVAL = 2

    old_transition_interval = transition_interval

    def showWindow():
        # _print('{} :: Showing window'.format(win_name))

        # win_handle = ctypes.windll.user32.FindWindowW(u'{}'.format(win_name), None)
        # print('win_handle: {}'.format(win_handle))
        # ctypes.windll.user32.ShowWindow(win_handle, 5)
        win_handle = win32gui.FindWindow(None, win_name)

        if other_win_name:
            # return
            #
            # _active_win_handle = int(sft_active_win_handle.value)
            # _active_win_name = win32gui.GetWindowText(_active_win_handle)
            #
            # print('_active_win_handle: {}'.format(_active_win_handle))
            # print('_active_win_name: {}'.format(_active_win_name))
            # print('win_name: {}'.format(win_name))
            #
            # win_handle = win32gui.FindWindow(None, win_name)
            #
            # if win_handle == _active_win_handle:
            #     return
            # try:
            #     win32gui.SetWindowPos(win_handle, _active_win_handle, 0, 0, 0, 0,
            #                           win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            # except BaseException as e:
            #     print('Failed {} --> {} : {}'.format(
            #         win_name, _active_win_name, e))
            #     # continue
            # else:
            #     print('showWindow :: {} --> {}'.format(
            #         win_name, _active_win_name))

            # win32gui.SetWindowPos(win_handle, win32con.HWND_BOTTOM, 0, 0, 0, 0,
            #                       win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            # win32gui.ShowWindow(win_handle, win32con.SW_SHOWNOACTIVATE)

            # win32gui.ShowWindow(win_handle, win32con.SW_RESTORE)
            loadImage(1)
            win32api.PostMessage(win_handle, win32con.WM_CHAR, 0x42, 0)
            return

        # print('win_handle: {}'.format(win_handle))
        win32gui.ShowWindow(win_handle, win32con.SW_RESTORE)
        keyboard.send('right')

        # if win_utils_available:
        #     winUtils.showWindow(win_name)
        # else:
        #     createWindow()

    def hideWindow():
        # _print('{} :: Hiding window'.format(win_name))
        # print('win_handle: {}'.format(win_handle))

        if other_win_name:
            # return
            # if other_win_name:
            #     win_handle = win32gui.FindWindow(None, win_name)
            #     _nazio_win_handle = win32gui.FindWindow(None, other_win_name)
            #
            #     print('win_handle: {}'.format(win_handle))
            #     print('_nazio_win_handle: {}'.format(_nazio_win_handle))
            #
            #     try:
            #         win32gui.SetWindowPos(win_handle, _nazio_win_handle, 0, 0, 0, 0,
            #                               win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            #     except BaseException as e:
            #         print('Failed {} --> {} : {}'.format(
            #             win_name, _active_win_name, e))
            #         # continue
            #     else:
            #         print('hideWindow :: {} --> {}'.format(
            #             win_name, _active_win_name))

            # win32gui.ShowWindow(win_handle, win32con.SW_MINIMIZE)

            return

        win_handle = win32gui.FindWindow(None, win_name)
        win32gui.ShowWindow(win_handle, win32con.SW_MINIMIZE)

        # win_handle = ctypes.windll.user32.FindWindow(u'{}'.format(win_name), None)
        # print('win_handle: {}'.format(win_handle))
        # ctypes.windll.user32.ShowWindow(win_handle, 0)
        # if win_utils_available:
        #     winUtils.hideWindow(win_name)
        # else:
        #     cv2.destroyWindow(win_name)

    def kb_callback(event):
        nonlocal set_wallpaper, n_images, wallpaper_mode, exit_program, borderless, img_id
        nonlocal old_transition_interval, transition_interval, reversed_pos, alpha, show_window

        # print('_params: {}'.format(_params))

        # _type = event.name
        # scan_code = event.scan_code
        # print('scan_code: {}'.format(scan_code))

        _type = event
        _print('hotkey: {}'.format(_type))

        if _type == 'ctrl+alt+esc':
            _print('exiting...')
            exit_program = 1
            if second_from_top:
                sft_exit_program.value = 1
            interrupt_wait.set()
        elif _type == 'ctrl+alt+right':
            # loadImage(1)
            interrupt_wait.set()
        elif _type == 'ctrl+alt+left':
            # loadImage(-1)
            if not video_mode:
                img_id[0] -= 2 * n_images
            interrupt_wait.set()
        elif _type == 'ctrl+alt+w':
            wallpaper_mode = 1 - wallpaper_mode
            if wallpaper_mode:
                set_wallpaper = 1
                _print('wallpaper mode enabled')
                cv2.destroyWindow(win_name)
                # minimizeWindow()
            else:
                _print('wallpaper mode disabled')
                createWindow()
                # maximizeWindow()
            interrupt_wait.set()
        elif _type == 'ctrl+alt+shift+w':
            wallpaper_mode = 1 - wallpaper_mode
            if wallpaper_mode:
                set_wallpaper = 2
                _print('wallpaper mode enabled')
                cv2.destroyWindow(win_name)
            else:
                _print('wallpaper mode disabled')
                createWindow()
            interrupt_wait.set()
        elif _type == 'ctrl+alt+=':
            n_images += 1
            loadImage(1, 1)
        elif _type == 'ctrl+alt+-':
            n_images -= 1
            if n_images < 1:
                n_images = 1
            loadImage(1, 1)
        elif _type == 'ctrl+alt+b':
            borderless = 1 - borderless
            if borderless:
                _print('Borderless stitching enabled')
            else:
                _print('Borderless stitching disabled')
        elif _type == 'ctrl+alt+$':
            n_images = 4
            loadImage(1, 1)
        elif _type == 'ctrl+alt+^':
            n_images = 6
            loadImage(1, 1)
        elif _type == 'ctrl+alt+!':
            n_images = 1
            loadImage(1, 1)
        elif _type == 'ctrl+alt+@':
            n_images = 2
            loadImage(1, 1)
        elif _type == 'ctrl+alt+up':
            transition_interval += 1 if transition_interval < 10 else 5

            # if transition_interval == MAX_TRANSITION_INTERVAL:
            #     transition_interval = old_transition_interval
            # else:
            #     old_transition_interval = transition_interval
            #     transition_interval = MAX_TRANSITION_INTERVAL
            _print('Setting transition interval to: {}'.format(transition_interval))
            if not video_mode:
                img_id[0] -= n_images
            interrupt_wait.set()
        elif _type == 'ctrl+alt+down':
            transition_interval -= 1 if transition_interval <= 10 else 5
            if transition_interval < 0:
                transition_interval = 0
            # if transition_interval == MIN_TRANSITION_INTERVAL:
            #     transition_interval = old_transition_interval
            # else:
            #     old_transition_interval = transition_interval
            #     transition_interval = MIN_TRANSITION_INTERVAL
            _print('Setting transition interval to: {}'.format(transition_interval))
            if not video_mode:
                img_id[0] -= n_images
            interrupt_wait.set()
        elif _type == 'ctrl+alt+p':
            reversed_pos = (reversed_pos + 1) % 3
            if not video_mode:
                img_id[0] -= n_images
            interrupt_wait.set()
        elif _type == 'ctrl+alt+a':
            alpha -= 0.1
            if alpha < 0:
                alpha = 1
            if not video_mode:
                img_id[0] -= n_images
            interrupt_wait.set()
        elif _type == 'ctrl+alt+shift+a':
            alpha += 0.1
            if alpha > 1:
                alpha = 0
            if not video_mode:
                img_id[0] -= n_images
            interrupt_wait.set()
        elif _type == 'play/pause media' or _type == -179:
            show_window = 1 - show_window
            if show_window:
                showWindow()
            else:
                hideWindow()
        # elif _type == 'previous track' or _type == -177:
        #     print('sending shift+left')
        #     keyboard.send('shift+left')
        # elif _type == 'next track' or _type == -176:
        #     print('sending shift+right')
        #     keyboard.send('shift+right')
        elif _type == 'ctrl+alt+0' or _type == 'ctrl+alt+)':
            if n_images == 1:
                _print('"' + os.path.abspath(img_fname) + '"')
            else:
                _print()
                for _idx in stack_idx:
                    if not video_mode:
                        _print('"' + os.path.abspath(img_fnames[_idx]) + '"')
                _print()

    hotkeys = [
        'ctrl+alt+esc',
        'ctrl+alt+right',
        'ctrl+alt+left',
        'ctrl+alt+w',
        'ctrl+alt+shift+w',
        'ctrl+alt+=',
        'ctrl+alt+-',
        'ctrl+alt+b',
        'ctrl+alt+$',
        'ctrl+alt+^',
        'ctrl+alt+!',
        'ctrl+alt+@',
        'ctrl+alt+up',
        'ctrl+alt+down',
        'ctrl+alt+0',
        'ctrl+alt+p',
        'ctrl+alt+a',
        'ctrl+alt+shift+a',
        -179,
        # -177,
        # -176,
    ]

    def add_hotkeys():
        # keyboard.on_press(kb_callback)
        for key in hotkeys:
            keyboard.add_hotkey(key, kb_callback, args=(key,))

    def remove_hotkeys():
        for key in hotkeys:
            keyboard.remove_hotkey(key)

    # if hotkeys_available:
    #     def handle_win_f3():
    #         print('Minimizing window')
    #         win_handle = ctypes.windll.user32.FindWindowW(None, win_name)
    #         ctypes.windll.user32.ShowWindow(win_handle, 6)
    #
    #
    #     def handle_win_f4():
    #         print('Restoring window')
    #         win_handle = ctypes.windll.user32.FindWindowW(None, win_name)
    #         ctypes.windll.user32.ShowWindow(win_handle, 9)
    #
    #
    #     HOTKEY_ACTIONS = {
    #         1: handle_win_f3,
    #         2: handle_win_f4
    #     }

    def moveWindow(_monitor_id, _win_name, _reversed_pos):
        nonlocal frg_positions
        if frg_win_titles:
            cv2.moveWindow(_win_name, frg_positions[frg_win_id][0], frg_positions[frg_win_id][1])
            return

        if mode == 0:
            _curr_monitor = _monitor_id
        elif mode == 1:
            if move_to_right:
                _curr_monitor = 4
            else:
                _curr_monitor = 2

        _y_offset = win_offset_y + monitors[_curr_monitor][1]

        if _reversed_pos == 0:
            cv2.moveWindow(_win_name, win_offset_x + monitors[_curr_monitor][0],
                           _y_offset)
        elif _reversed_pos == 1:
            cv2.moveWindow(_win_name,
                           int(win_offset_x + monitors[_curr_monitor][0] + (width - dst_img.shape[1]) / 2),
                           _y_offset)
        elif _reversed_pos == 2:
            cv2.moveWindow(_win_name, win_offset_x + int(monitors[_curr_monitor][0] + width - dst_img.shape[1]),
                           _y_offset)

    img_id[0] += n_images - 1
    loadImage(set_grid_size=set_grid_size)
    exit_program = 0

    numpad_to_ascii = {
        2293760: '1',
        2228224: '2',
        2359296: '3',
        2162688: '4',
    }
    images_to_sort = {}
    images_to_sort_inv = {}

    if enable_hotkeys:
        _print('Hotkeys are enabled')
        add_hotkeys()

    # elif not on_top and second_from_top:
    #     keyboard.add_hotkey('ctrl+alt+shift+a', mouse_click_callback, args=(key,))

    def second_from_top_callback(
            # x, y, button, pressed
    ):
        nonlocal prev_active_handle, prev_active_win_name, active_monitor_id

        # print('button: {}'.format(button))
        # print('pressed: {}'.format(pressed))

        active_handle = win32gui.GetForegroundWindow()
        active_win_name = win32gui.GetWindowText(active_handle)
        # print('active_win_name: {}'.format(active_win_name))

        if active_win_name and (prev_active_handle is None or prev_active_handle != active_handle) and \
                active_win_name not in [win_name, ] + dup_win_names and \
                all([k not in active_win_name for k in sft_exceptions]) and \
                all([any([k1 not in active_win_name for k1 in k]) for k in sft_exceptions_multi]):
            prev_active_handle = active_handle
            prev_active_win_name = active_win_name

            rect = win32gui.GetWindowRect(active_handle)
            x = (rect[0] + rect[2]) / 2.0
            y = (rect[1] + rect[3]) / 2.0

            _monitor_id = 0
            min_dist = np.inf
            for curr_id, monitor in enumerate(monitors):
                _centroid_x = (monitor[0] + monitor[0] + 1920) / 2.0
                _centroid_y = (monitor[1] + monitor[1] + 1080) / 2.0
                dist = (x - _centroid_x) ** 2 + (y - _centroid_y) ** 2
                if dist < min_dist:
                    min_dist = dist
                    _monitor_id = curr_id

            # print('active_win_name: {} with pos: {} on monitor {}'.format(active_win_name, rect, _monitor_id))

            if _monitor_id == monitor_id:
                _win_handle = win32gui.FindWindow(None, win_name)
                active_monitor_id = _monitor_id
                win32api.PostMessage(_win_handle, win32con.WM_CHAR, 0x42, 0)
                # print('temp: {}'.format(temp))

                # win32gui.ShowWindow(_win_handle, 5)
                # win32gui.SetForegroundWindow(_win_handle)
                #
                # win32gui.ShowWindow(win_handle, 5)
                # win32gui.SetForegroundWindow(win_handle)

            elif duplicate_window and _monitor_id in dup_monitor_ids:
                _i = dup_monitor_ids.index(_monitor_id)
                if second_from_top > _i + 1:
                    _win_handle = win32gui.FindWindow(None, dup_win_names[_i])
                    active_monitor_id = _monitor_id
                    win32api.PostMessage(_win_handle, win32con.WM_CHAR, 0x44, 0)
                    # print('temp: {}'.format(temp))

                    # win32gui.ShowWindow(_win_handle, 5)
                    # win32gui.SetForegroundWindow(_win_handle)
                    #
                    # win32gui.ShowWindow(win_handle, 5)
                    # win32gui.SetForegroundWindow(win_handle)

    # def second_from_top_fn():
    #     while second_from_top and not exit_program:
    #         time.sleep(1)
    #         second_from_top_callback()

    # class StoppableThread(threading.Thread):
    #     """Thread class with a stop() method. The thread itself has to check
    #     regularly for the stopped() condition."""
    #
    #     def __init__(self, *args, **kwargs):
    #         super(StoppableThread, self).__init__(*args, **kwargs)
    #         self._stop_event = threading.Event()
    #
    #     def stop(self):
    #         self._stop_event.set()
    #
    #     def stopped(self):
    #         return self._stop_event.is_set()
    #
    #
    # class MyTask(StoppableThread):
    #     def run(self):
    #         while not self.stopped():
    #             time.sleep(1)
    #             second_from_top_callback()

    # active_win_info = [None, None, None]
    # active_monitor_id, active_win_name, active_win_handle = active_win_info

    # if sft_vars is not None:
    #     sft_active_monitor_id, sft_active_win_handle, sft_active_monitor_id_2, sft_active_win_handle_2 = sft_vars
    #     sft_other_vars = sft_active_monitor_id_2, sft_active_win_handle_2, other_win_name
    # else:

    sft_active_monitor_id = multiprocessing.Value('I', lock=False)
    sft_active_win_handle = multiprocessing.Value('L', lock=False)

    # sft_other_vars = None

    sft_exit_program = multiprocessing.Value('L', 0, lock=False)
    # sft_active_win_name = multiprocessing.Value(ctypes.c_char_p, lock=False)

    # manager = multiprocessing.Manager()
    # active_win_info = manager.dict()

    second_from_top_thread = None
    if second_from_top:
        second_from_top_thread = Process(target=sft.second_from_top_fn,
                                         args=(sft_active_monitor_id, sft_active_win_handle, sft_exit_program,
                                               second_from_top, monitors, win_name,
                                               dup_win_names, monitor_id, dup_monitor_ids,
                                               duplicate_window, only_maximized, frg_win_handles,
                                               # sft_other_vars
                                               ))

        # second_from_top_thread = threading.Thread(target=second_from_top_fn)
        # second_from_top_thread = MyTask()

        second_from_top_thread.start()

        # mouse.on_click(second_from_top_callback, args=())
        # mouse.on_middle_click(second_from_top_callback, args=())
        # mouse.on_button(second_from_top_callback, buttons=('x', 'x2'), types=('down'))
        # mouse_listener = mouse.Listener(
        #     on_click=second_from_top_callback)
        # mouse_listener.start()

    if not show_window:
        hideWindow()

    first_img = True

    while not exit_program:
        # if video_mode:
        #     print('show_window: {}'.format(show_window))

        # if not show_window:
        #     interrupt_wait.wait(0.1)
        #     interrupt_wait.clear()
        #     continue

        # exit_program = kb_params[0]
        # wallpaper_mode = kb_params[1]
        # if exit_program:
        #     break

        if wallpaper_mode:
            interrupt_wait.wait(transition_interval)
            interrupt_wait.clear()
            loadImage(1)
            continue

        _row_offset, _col_offset = row_offset, col_offset
        if end_row + _row_offset > dst_height:
            _row_offset -= end_row + row_offset - dst_height

        if _row_offset + start_row < 0:
            _row_offset = -start_row

        if end_col + _col_offset > dst_width:
            _col_offset -= end_col + _col_offset - dst_width

        if _col_offset + start_col < 0:
            _col_offset = -start_col

        temp = src_img_ar[int(start_row + _row_offset):int(end_row + _row_offset),
               int(start_col + _col_offset):int(end_col + _col_offset), :]

        try:
            dst_img = cv2.resize(temp, (width, height))
        except cv2.error as e:
            _print('Resizing error: {}'.format(e))
            temp_height, temp_width, _ = temp.shape
            _print('temp_height: ', temp_height)
            _print('temp_width: ', temp_width)
            if temp_height:
                temp_aspect_ratio = float(temp_width) / float(temp_height)
                _print('temp_aspect_ratio: ', temp_aspect_ratio)
            _print('_col_offset: ', _col_offset)
            _print('_row_offset: ', _row_offset)

        # if mode == 0 and not fullscreen:
        if not fullscreen:
            temp_height, temp_width, _ = temp.shape
            temp_height_ratio = float(temp_height) / float(height)

            win_start_row = int(max(0, src_start_row - start_row) / temp_height_ratio)
            win_end_row = height - int(max(0, end_row - src_end_row) / temp_height_ratio)

            win_start_col = int(max(0, src_start_col - start_col) / temp_height_ratio)
            win_end_col = width - int(max(0, end_col - src_end_col) / temp_height_ratio)

            dst_img = dst_img[win_start_row:win_end_row, win_start_col:win_end_col, :]

            if frg_win_titles and not first_img:
                hwnd = frg_win_handles[frg_win_id]
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                except:
                    pass
                else:
                    if DwmGetWindowAttribute:
                        ext_rect = ctypes.wintypes.RECT()
                        DWMWA_EXTENDED_FRAME_BOUNDS = 9
                        DwmGetWindowAttribute(
                            ctypes.wintypes.HWND(hwnd),
                            ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
                            ctypes.byref(ext_rect),
                            ctypes.sizeof(ext_rect)
                        )
                        border = [ext_rect.left - rect[0], ext_rect.top - rect[1],
                                  rect[2] - ext_rect.right, rect[3] - ext_rect.bottom]
                        # rect = [rect[0] - border[0], rect[1] - border[1],
                        #         rect[2] + border[0] + border[2], rect[3] + border[1] + border[3]]

                        rect = [rect[0] + border[0], rect[1] + border[1],
                                rect[2] - border[2], rect[3] - border[3]]

                    frg_positions[frg_win_id] = rect
                x1, y1, x2, y2 = frg_positions[frg_win_id]
                reversed_pos = frg_reversed_pos[frg_win_id]
                __w, __h = x2 - x1, y2 - y1
                dst_img = resizeAR(dst_img, __w, __h, placement_type=reversed_pos)
                # print('__w: ', __w)
                # print('__h: ', __h)

            # print(':: reversed_pos: ', reversed_pos)

            if first_img:
                first_img = False

            moveWindow(monitor_id, win_name, reversed_pos)
            # if mode == 0:
            #     _curr_monitor = monitor_id
            # elif mode == 1:
            #     if move_to_right:
            #         _curr_monitor = 4
            #     else:
            #         _curr_monitor = 2
            #
            # _y_offset = win_offset_y + monitors[_curr_monitor][1]
            #
            # if reversed_pos == 0:
            #
            #     cv2.moveWindow(win_name, win_offset_x + monitors[_curr_monitor][0],
            #                    _y_offset)
            # elif reversed_pos == 1:
            #     cv2.moveWindow(win_name,
            #                    int(win_offset_x + monitors[_curr_monitor][0] + (width - dst_img.shape[1]) / 2),
            #                    _y_offset)
            # elif reversed_pos == 2:
            #     cv2.moveWindow(win_name, win_offset_x + int(monitors[_curr_monitor][0] + width - dst_img.shape[1]),
            #                    _y_offset)

            if duplicate_window:
                for _i, _win_name2 in enumerate(dup_win_names):
                    moveWindow(dup_monitor_ids[_i], _win_name2, dup_reversed_pos[_i])

            # if win_utils_available:
            #     winUtils.hideBorder2(win_name, on_top)

        cv2.imshow(win_name, dst_img)

        # def callback(hwnd, extra):
        #     rect = win32gui.GetWindowRect(hwnd)
        #     x = rect[0]
        #     y = rect[1]
        #     w = rect[2] - x
        #     h = rect[3] - y
        #     print("Window %s:" % win32gui.GetWindowText(hwnd))
        #     print("\tLocation: (%d, %d)" % (x, y))
        #     print("\t    Size: (%d, %d)" % (w, h))
        #
        # win32gui.EnumWindows(callback, None)

        # winUtils.setBehindTopMost(win_name)

        if duplicate_window:
            for _i, _win_name2 in enumerate(dup_win_names):
                cv2.imshow(_win_name2, dst_img)
            # winUtils.setBehindTopMost(dup_win_names)

        # if win_utils_available:
        #     winUtils.loseFocus(win_name)

        # winUtils.hideBorder2(win_name, on_top)
        # winUtils.show2(win_name)

        # if win_utils_available:
        #     winUtils.show(win_name, dst_img, 0)
        # else:
        #     cv2.imshow(win_name, dst_img)

        # active_win_name = win32gui.GetWindowText(win32gui.GetForegroundWindow())
        # print('active_win_name: {}'.format(active_win_name))
        #
        # if (prev_active_win_name is None or prev_active_win_name != active_win_name) and active_win_name not in (
        #         win_name, dup_win_names):
        #     prev_active_win_name = active_win_name
        #
        #     win_handle = win32gui.FindWindow(None, active_win_name)
        #     rect = win32gui.GetWindowRect(win_handle)
        #     x = (rect[0] + rect[2]) / 2.0
        #     y = (rect[1] + rect[3]) / 2.0
        #
        #     _monitor_id = 0
        #     min_dist = np.inf
        #     for curr_id, monitor in enumerate(monitors):
        #         _centroid_x = (monitor[0] + monitor[0] + 1920) / 2.0
        #         _centroid_y = (monitor[1] + monitor[1] + 1080) / 2.0
        #         dist = (x - _centroid_x) ** 2 + (y - _centroid_y) ** 2
        #         if dist < min_dist:
        #             min_dist = dist
        #             _monitor_id = curr_id
        #
        #     print('active_win_name: {} with pos: {} on monitor {}'.format(active_win_name, rect, _monitor_id))
        #
        #     if _monitor_id == monitor_id:
        #         _win_handle = win32gui.FindWindow(None, win_name)
        #         # win32gui.ShowWindow(_win_handle, 5)
        #         win32gui.SetForegroundWindow(_win_handle)
        #         # win32gui.ShowWindow(win_handle, 5)
        #         win32gui.SetForegroundWindow(win_handle)
        #
        #     elif _monitor_id == dup_monitor_ids:
        #         _win_handle = win32gui.FindWindow(None, dup_win_names)
        #         # win32gui.ShowWindow(_win_handle, 5)
        #         win32gui.SetForegroundWindow(_win_handle)
        #         # win32gui.ShowWindow(win_handle, 5)
        #         win32gui.SetForegroundWindow(win_handle)

        if not show_window:
            k = cv2.waitKeyEx(0)
        elif speed == 0 and auto_progress:
            if video_mode:
                k = cv2.waitKeyEx(transition_interval)
            else:
                k = cv2.waitKeyEx(transition_interval * 1000)
        else:
            k = cv2.waitKeyEx(1)

        # k = cv2.waitKeyEx(0)
        # k = -1

        # if is_switching:
        #     print('Ignoring key input while switching')
        if k < 0:
            auto_progress_type = 1
        else:
            # print('k: {}'.format(k))
            if k == 27 or end_exec:
                exit_program = 1
                _exit_neatly()
                break
            elif k == 13:
                changeMode()
            elif k == 10:
                if mode == 1:
                    widescreen_mode = 1 - widescreen_mode
                else:
                    move_to_right = 1 - move_to_right
                changeMode()
            elif k == ord('g'):
                # grid transpose
                grid_size = (grid_size[1], grid_size[0])
                loadImage()
            elif k == ord('C'):
                # single column grid
                grid_size = (n_images, 1)
                loadImage()
            elif k == ord('R'):
                # single row grid
                grid_size = (1, n_images)
                loadImage()
            elif k == ord('h'):
                show_window = 1 - show_window
                if show_window:
                    # _print('{} :: showing window\n'.format(win_name))
                    showWindow()
                else:
                    # _print('{} :: hiding window\n'.format(win_name))
                    hideWindow()
            elif k == ord('r'):
                if video_mode:
                    _print('Reversing video')
                    for _id in img_id:
                        img_id[_id] = total_frames[_id] - img_id[_id] - 1
                        src_files[_id] = list(reversed(src_files[_id]))
                else:
                    random_mode = 1 - random_mode
                    if random_mode:
                        _print('Random mode enabled')
                        for _id in img_id:
                            src_files_rand[_id] = list(np.random.permutation(src_files[_id]))
                            # img_id[_id] = src_files_rand[_id].index(img_fname)
                            img_id[_id] = 0
                    else:
                        _print('Random mode disabled')
                        if not video_mode:
                            for _load_id in range(n_images):
                                src_id = _load_id % n_src
                                img_id[src_id] = src_files[src_id].index(img_fnames[_load_id])
                                # img_id[_id] = 0
            elif k == ord('c'):
                auto_progress = 1 - auto_progress
                if auto_progress:
                    _print('Auto progression enabled')
                else:
                    _print('Auto progression disabled')
            elif k == ord('Q'):
                auto_progress_video = 1 - auto_progress_video
                if auto_progress_video:
                    _print('Video auto progression enabled')
                else:
                    _print('Video auto progression disabled')
            elif k == ord('e'):
                reverse_video = 1 - reverse_video
                if reverse_video:
                    _print('Video reversal enabled')
                else:
                    _print('Video reversal disabled')
            elif k == ord('q'):
                # if video_mode:
                rotate_images += 1
                if rotate_images > 3:
                    rotate_images = 0
                _print('Rotating video by {} degrees'.format(rotate_images * 90))
                src_images = []
                loadImage()
                # else:
                #     random_mode = 1 - random_mode
                #     if random_mode:
                #         print('Random mode enabled')
                #         src_file_list_rand = list(np.random.permutation(src_file_list))
                #     else:
                #         print('Random mode disabled')
                #     auto_progress = 1 - auto_progress
                #     if auto_progress:
                #         print('Auto progression enabled')
                #     else:
                #         print('Auto progression disabled')
            elif k == ord('b'):
                borderless = 1 - borderless
                if borderless:
                    _print('Borderless stitching enabled')
                else:
                    _print('Borderless stitching disabled')
                loadImage()
            elif k == ord('n'):
                max_switches -= 1
                if max_switches < 1:
                    max_switches = 1
            elif k == ord('d'):
                duplicate_window = 1 - duplicate_window
                if duplicate_window:
                    for _i, _win_name2 in enumerate(dup_win_names):
                        createWindow(_win_name2)
                    _print('{} duplicate windows enabled'.format(len(_win_name2)))
                else:
                    for _i, _win_name2 in enumerate(dup_win_names):
                        cv2.destroyWindow(_win_name2)
                    _print('duplicate window disabled')
            elif k == ord('N'):
                max_switches += 1
            elif ord('1') <= k <= ord('5'):
                _monitor_id = k - ord('1')
                try:
                    active_win_name = win32gui.GetWindowText(win32gui.GetForegroundWindow())
                    # success = win32gui.MoveWindow(
                    #     win32gui.GetForegroundWindow(),
                    #     int(win_offset_x + monitors[monitor_id][0]),
                    #     int(win_offset_y + monitors[monitor_id][1]),
                    #     int(dst_img.shape[1]),
                    #     int(dst_img.shape[0]),
                    #     1
                    # )
                    # success = 0
                    # if not success:
                    #     print('Move failed')
                except:
                    active_win_name = win_name

                if active_win_name == win_name:
                    monitor_id = _monitor_id
                elif active_win_name in dup_win_names:
                    _i = dup_win_names.index(active_win_name)
                    dup_monitor_ids[_i] = _monitor_id
                _print('moving window {}'.format(active_win_name))
                cv2.moveWindow(active_win_name, win_offset_x + monitors[_monitor_id][0],
                               win_offset_y + monitors[_monitor_id][1])
                # createWindow()
            # elif k == ord('2'):
            #     monitor_id = 1
            #     cv2.moveWindow(win_name, win_offset_x + monitors[1][0], win_offset_y + monitors[1][1])
            #     # createWindow()
            # elif k == ord('3'):
            #     monitor_id = 2
            #     cv2.moveWindow(win_name, win_offset_x + monitors[2][0], win_offset_y + monitors[2][1])
            #     # createWindow()
            # elif k == ord('4'):
            #     monitor_id = 3
            #     cv2.moveWindow(win_name, win_offset_x + monitors[3][0], win_offset_y + monitors[3][1])
            #     # createWindow()
            # elif k == ord('5'):
            #     monitor_id = 4
            #     cv2.moveWindow(win_name, win_offset_x + monitors[4][0], win_offset_y + monitors[4][1])
            #     # createWindow()
            elif k == 32:
                is_paused = 1 - is_paused
                if speed == 0:
                    speed = old_speed
                else:
                    old_speed = speed
                    speed = 0
            elif k == ord('P'):
                try:
                    active_win_name = win32gui.GetWindowText(win32gui.GetForegroundWindow())
                except:
                    active_win_name = win_name

                if active_win_name == win_name:
                    reversed_pos -= 1
                    if reversed_pos < 0:
                        reversed_pos = 2
                    # print('reversed_pos: ', reversed_pos)
                    if fullscreen or mode == 1:
                        loadImage(0)
                    elif not reversed_pos:
                        cv2.moveWindow(win_name, win_offset_x + monitors[monitor_id][0],
                                       win_offset_y + monitors[monitor_id][1])
                elif active_win_name in dup_win_names:
                    _i = dup_win_names.index(active_win_name)
                    _monitor_id2 = dup_monitor_ids[_i]
                    _dup_reversed_pos = dup_reversed_pos[_i]
                    _dup_reversed_pos -= 1
                    dup_reversed_pos[_i] = _dup_reversed_pos
                    if _dup_reversed_pos < 0:
                        _dup_reversed_pos = 2
                    if fullscreen or mode == 1:
                        loadImage(0)
                    elif not _dup_reversed_pos:
                        cv2.moveWindow(active_win_name, win_offset_x + monitors[_monitor_id2][0],
                                       win_offset_y + monitors[_monitor_id2][1])
            elif k == ord('p'):
                try:
                    active_win_name = win32gui.GetWindowText(win32gui.GetForegroundWindow())
                except:
                    active_win_name = win_name

                if active_win_name == win_name:
                    reversed_pos = (reversed_pos + 1) % 3
                    # print('reversed_pos: ', reversed_pos)
                    if fullscreen or mode == 1:
                        loadImage(0)
                    elif not reversed_pos:
                        cv2.moveWindow(active_win_name, win_offset_x + monitors[monitor_id][0],
                                       win_offset_y + monitors[monitor_id][1])
                elif active_win_name in dup_win_names:
                    _i = dup_win_names.index(active_win_name)
                    _monitor_id2 = dup_monitor_ids[_i]
                    _dup_reversed_pos = dup_reversed_pos[_i]
                    _dup_reversed_pos = (_dup_reversed_pos + 1) % 3
                    dup_reversed_pos[_i] = _dup_reversed_pos

                    # print('reversed_pos: ', reversed_pos)
                    if fullscreen or mode == 1:
                        loadImage(0)
                    elif not _dup_reversed_pos:
                        cv2.moveWindow(active_win_name, win_offset_x + monitors[_monitor_id2][0],
                                       win_offset_y + monitors[_monitor_id2][1])

            elif k == ord('B'):
                # winUtils.setBehindTopMost(win_name, prev_active_win_name)

                # cv2.destroyWindow(win_name)
                # win_name += 'k'
                # createWindow(win_name)
                # on_top = 1
                # hideBorder(win_name)

                # active_monitor_id, active_win_name, active_win_handle = active_win_info

                if not show_window:
                    continue

                _active_monitor_id = int(sft_active_monitor_id.value)
                _active_win_handle = int(sft_active_win_handle.value)

                if frg_win_titles:
                    try:
                        frg_win_id = frg_win_handles.index(_active_win_handle)
                    except ValueError:
                        pass

                _active_win_name = win32gui.GetWindowText(_active_win_handle)
                # _active_win_name = sft_active_win_name.value.decode("utf-8")

                # print('vwm: _active_win_handle: {}'.format(_active_win_handle))

                win_handle = win32gui.FindWindow(None, win_name)

                # active_win_handle = prev_active_handle
                # active_win_handle = win32gui.FindWindow(None, prev_active_win_name)
                # print('_win_handle: {}'.format(_win_handle))
                # print('active_win_handle: {}'.format(active_win_handle))

                # while True:
                # is_switching = 1
                try:
                    # win32gui.SetForegroundWindow(win_handle)
                    # win32gui.SetFocus(win_handle)

                    win32gui.SetWindowPos(win_handle, _active_win_handle, 0, 0, 0, 0,
                                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                    """calling once doesn't always work"""
                    win32gui.SetWindowPos(win_handle, _active_win_handle, 0, 0, 0, 0,
                                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                    # while True:
                    #     win32gui.SetWindowPos(win_handle, _active_win_handle, 0, 0, 0, 0,
                    #                           win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                    #     user32 = ctypes.windll.user32
                    #     _next = user32.GetWindow(_active_win_handle, win32con.GW_HWNDNEXT)
                    #     if _next != win_handle:
                    #         _next_name = win32gui.GetWindowText(_next)
                    #
                    #         print('second from top window: {} ({}) does not match the desired one: {} ({})'.format(
                    #             _next, _next_name, win_handle, win_name
                    #         ))
                    #     else:
                    #         print('second from top window: {} ({}) matches the desired one: {} ({})'.format(
                    #             _next, _next_name, win_handle, win_name
                    #         ))
                    #         break
                except BaseException as e:
                    _print('Failed {} --> {} : {}'.format(
                        win_name, prev_active_win_name, e))
                    # continue
                else:
                    prev_active_win_name = _active_win_name

                    _print('{} --> {}'.format(
                        win_name, prev_active_win_name))
                # time.sleep(1)
                # is_switching = 0

                # try:
                #     win32gui.SetForegroundWindow(active_win_handle)
                #     # win32gui.SetFocus(active_win_handle)
                # except BaseException as e:
                #     print('Failed to change window status for {} : {}'.format(prev_active_win_name, e))
                #     continue
                # print('Successfully Changed window status for {}'.format(prev_active_win_name))

                # break
                # on_top = 0
                # hideBorder(win_name)

            elif k == ord('D'):

                if not show_window:
                    continue
                # winUtils.setBehindTopMost(dup_win_names, prev_active_win_name)

                # cv2.destroyWindow(dup_win_names)
                # dup_win_names += 'k'
                # createWindow(dup_win_names)

                # on_top = 1
                # hideBorder(dup_win_names)

                _active_monitor_id = int(sft_active_monitor_id.value)
                _active_win_handle = int(sft_active_win_handle.value)

                if frg_win_titles:
                    try:
                        frg_win_id = frg_win_handles.index(_active_win_handle)
                    except ValueError:
                        pass

                _active_win_name = win32gui.GetWindowText(_active_win_handle)

                # _active_win_name = sft_active_win_name.value.decode("utf-8")

                # print('vwm: _active_win_handle: {}'.format(_active_win_handle))

                # active_monitor_id, active_win_name, active_win_handle = active_win_info

                try:
                    _i = dup_monitor_ids.index(_active_monitor_id)
                except ValueError as e:
                    _print('Window switching failed: {}'.format(e))
                else:
                    win_handle = win32gui.FindWindow(None, dup_win_names[_i])

                    # active_win_handle = win32gui.GetForegroundWindow()
                    # active_win_name = win32gui.GetWindowText(active_win_handle)

                    prev_active_win_name = _active_win_name

                    # active_win_handle = prev_active_handle

                    # active_win_handle = win32gui.FindWindow(None, prev_active_win_name)
                    # print('_win_handle: {}'.format(_win_handle))
                    # print('active_win_handle: {}'.format(active_win_handle))

                    # while True:

                    # is_switching = 1

                    try:
                        # win32gui.SetForegroundWindow(win_handle)
                        # win32gui.SetFocus(win_handle)
                        win32gui.SetWindowPos(win_handle, _active_win_handle, 0, 0, 0, 0,
                                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                        # win32gui.SetWindowPos(win_handle, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                        #                       win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                    except BaseException as e:
                        _print('Failed {} --> {} : {}'.format(
                            dup_win_names[_i], prev_active_win_name, e))
                        # continue
                    else:
                        _print('{} --> {}'.format(
                            dup_win_names[_i], prev_active_win_name))

                    # time.sleep(1)
                    # is_switching = 0

                    # try:
                    #     win32gui.SetForegroundWindow(active_win_handle)
                    #     # win32gui.SetFocus(active_win_handle)
                    # except BaseException as e:
                    #     print('Failed to change window status for {} : {}'.format(prev_active_win_name, e))
                    #     continue
                    # print('Successfully Changed window status for {}'.format(prev_active_win_name))

                    # break

                    # on_top = 0
                    # hideBorder(dup_win_names)

            elif k == ord('V'):
                if second_from_top:
                    second_from_top = 0
                    _print('second_from_top disabled')
                    sys.stdout.write('waiting for second_from_top_thread to exit...')
                    sft_exit_program.value = 1
                    # second_from_top_thread.terminate()
                    sys.stdout.write('done\n')
                else:
                    second_from_top = 1
                    _print('second_from_top enabled')
                    sft_exit_program.value = 0
                    second_from_top_thread = Process(target=sft.second_from_top_fn,
                                                     args=(
                                                         sft_active_monitor_id, sft_active_win_handle, sft_exit_program,
                                                         second_from_top, monitors, win_name,
                                                         dup_win_names, monitor_id, dup_monitor_ids,
                                                         duplicate_window, only_maximized, frg_win_handles,
                                                         # sft_other_vars
                                                     ))
                    second_from_top_thread.start()
                    # mouse.on_click(second_from_top_callback, args=())
                    # mouse.unhook(second_from_top_callback)
            elif k == ord('v'):
                on_top = 1 - on_top
                hideBorder(win_name, on_top)
                if duplicate_window:
                    for _win_name2 in dup_win_names:
                        hideBorder(_win_name2, on_top)
            elif k == ord('t'):
                transition_interval -= transition_interval_diff
                if transition_interval < 1:
                    transition_interval = 1
                _print('Setting transition interval to: {}'.format(transition_interval))
            elif k == ord('T'):
                transition_interval += transition_interval_diff
                _print('Setting transition interval to: {}'.format(transition_interval))
            elif k == ord('m') or k == ord('M'):
                minimizeWindow()
            elif k == ord('W'):
                set_wallpaper = 0 if set_wallpaper else 2
                if set_wallpaper:
                    minimizeWindow()
                    loadImage()
            elif k == ord('w'):
                set_wallpaper = 0 if set_wallpaper else 1
                if set_wallpaper:
                    minimizeWindow()
                    loadImage()
            # elif k == ord('e') or k == ord('E'):
            #     wallpaper_mode = 1 - wallpaper_mode
            #     if wallpaper_mode:
            #         set_wallpaper = 2 if k == ord('E') else 1
            #         _print('wallpaper mode enabled')
            #         cv2.destroyWindow(win_name)
            #         add_hotkeys()
            #     else:
            #         _print('wallpaper mode disabled')
            #         createWindow(win_name)
            #         if duplicate_window:
            #             for _win_name2 in dup_win_names:
            #                 createWindow(_win_name2)
            #         remove_hotkeys()
            elif k == ord(','):
                height -= 5
                if height < 10:
                    height = 10
                loadImage()
            elif k == ord('.'):
                height += 5
                loadImage()
            elif k == ord('<'):
                width -= 5
                if width < 10:
                    width = 10
                loadImage()
            elif k == ord('>'):
                width += 5
                loadImage()
            elif k == ord('/'):
                height = _height
                loadImage()
            elif k == ord('?'):
                width = _width
                loadImage()
            elif k == ord('+'):
                n_images += 1
                loadImage(1, 1, 1)
            elif k == ord('-'):
                n_images -= 1
                if n_images < 1:
                    n_images = 1

                loadImage(1, 1)
            elif k == ord('='):
                predef_n_image_id = (predef_n_image_id + 1) % n_predef_n_images
                n_images = predef_n_images[predef_n_image_id]
                loadImage(1, 1, 1)
            elif k == ord('_'):
                predef_n_image_id -= 1
                if predef_n_image_id < 0:
                    predef_n_image_id = n_predef_n_images - 1
                n_images = predef_n_images[predef_n_image_id]
                loadImage(1, 1, 1)
            elif k == ord('!'):
                predef_n_image_id = 0
                n_images = predef_n_images[predef_n_image_id]
                loadImage(1, 1, 1)
            elif k == ord('@'):
                predef_n_image_id = 1
                n_images = predef_n_images[predef_n_image_id]
                loadImage(1, 1, 1)
            elif k == ord('#'):
                predef_n_image_id = 2
                n_images = predef_n_images[predef_n_image_id]
                loadImage(1, 1, 1)
            elif k == ord('$'):
                predef_n_image_id = 3
                n_images = predef_n_images[predef_n_image_id]
                loadImage(1, 1, 1)
            elif k == ord('%'):
                predef_n_image_id = 4
                n_images = predef_n_images[predef_n_image_id]
                loadImage(1, 1, 1)
            elif k == ord('^'):
                predef_n_image_id = 5
                n_images = predef_n_images[predef_n_image_id]
                loadImage(1, 1, 1)
            elif k == ord('&'):
                predef_n_image_id = 6
                n_images = predef_n_images[predef_n_image_id]
                loadImage(1, 1, 1)
            elif k == ord('*'):
                predef_n_image_id = 7
                n_images = predef_n_images[predef_n_image_id]
                loadImage(1, 1, 1)
            elif k == ord('('):
                predef_n_image_id = 8
                n_images = predef_n_images[predef_n_image_id]
                loadImage(1, 1, 1)
            elif k == ord(')'):
                predef_n_image_id = 9
                n_images = predef_n_images[predef_n_image_id]
                loadImage(1, 1, 1)
            elif k == ord('i'):
                direction = -direction
            elif k == ord('s') or k == ord('l') or k == ord('R'):
                loadImage()
            elif k == 2490368:
                # up
                updateZoom(_speed=old_speed, _direction=-1)
            elif k == 2621440:
                # down
                updateZoom(_speed=old_speed, _direction=1)
            elif k == ord('a'):
                alpha -= 0.1
                if alpha < 0:
                    alpha = 1
                loadImage(0)
            elif k == ord('A'):
                alpha += 0.1
                if alpha > 1:
                    alpha = 0
                loadImage(0)
            elif k == 2555904 or k == 39:
                # right
                if video_mode and auto_progress:
                    vid_id = (vid_id + 1) % n_videos
                    src_path = video_files_list[vid_id]
                    loadVideo(0)
                    loadImage()
                else:
                    if speed == 0 and auto_progress:
                        auto_progress_type = 1
                    else:
                        loadImage(1)
            elif k == 2424832 or k == 40:
                # left
                if video_mode and auto_progress:
                    vid_id -= 1
                    if vid_id < 0:
                        vid_id = n_videos - 1
                    src_path = video_files_list[vid_id]
                    loadVideo(0)
                    loadImage()
                else:
                    if speed == 0 and auto_progress:
                        auto_progress_type = -1
                    else:
                        loadImage(-1)
            elif k == ord('F') or k == ord('0'):
                if n_images == 1:
                    _print('"' + os.path.abspath(img_fname) + '"')
                else:
                    _print()
                    for _idx in stack_idx:
                        if not video_mode:
                            _print('"' + os.path.abspath(img_fnames[_idx]) + '"')
                    _print()
            elif k == ord('f') or k == ord('/') or k == ord('?'):
                fullscreen = 1 - fullscreen
                createWindow(win_name)
                if duplicate_window:
                    for _win_name2 in dup_win_names:
                        createWindow(_win_name2)
                if fullscreen:
                    _print('fullscreen mode enabled')
                else:
                    _print('fullscreen mode disabled')
            else:
                try:
                    numpad_key = numpad_to_ascii[k]
                except KeyError as e:
                    _print('Unknown key: {} :: {}'.format(k, e))
                else:
                    sortImage(img_fname, numpad_key)

        # if hotkeys_available:
        #     msg = wintypes.MSG()
        #     if user32.GetMessageA(byref(msg), None, 0, 0) != 0:
        #         if msg.message == win32con.WM_HOTKEY:
        #             action_to_take = HOTKEY_ACTIONS.get(msg.wParam)
        #             if action_to_take:
        #                 action_to_take()

        # direction = motionStep(direction)

        updateZoom()

        if speed == 0 and auto_progress:
            if auto_progress_type == 0:
                pass
            else:
                # time.sleep(transition_interval)
                loadImage(auto_progress_type)
                # end_time = time.time()
                # if end_time - start_time >= transition_interval:
                #     loadImage(1)
                auto_progress_type = 0

        # print('end_row: ', end_row)
        # print('start_col: ', start_col)
        # print('end_col: ', end_col)

        # print('\n')

    if wallpaper_mode:
        remove_hotkeys()
    else:
        cv2.destroyWindow(win_name)

    # if second_from_top:
    #     sys.stdout.write('waiting for second_from_top_thread to exit...')
    #     second_from_top_thread.terminate()
    #     sys.stdout.write('done\n')

    if images_to_sort:
        for k in images_to_sort.keys():
            _print('k: ', k)
            for orig_file_path in images_to_sort[k]:
                if not os.path.isfile(orig_file_path):
                    continue
                _print('orig_file_path: ', orig_file_path)
                orig_file_path = os.path.abspath(orig_file_path)
                sort_dir = os.path.join(os.path.dirname(orig_file_path), k)
                if not os.path.isdir(sort_dir):
                    os.makedirs(sort_dir)
                sort_file_path = os.path.join(sort_dir, os.path.basename(orig_file_path))
                _print('Moving {} to {}'.format(orig_file_path, sort_file_path))
                shutil.move(orig_file_path, sort_file_path)

    if set_wallpaper:
        win_wallpaper_func(SPI_SETDESKWALLPAPER, 0, orig_wp_fname, 0)


# print('Here we are')

if __name__ == '__main__':
    # print('Here we are')
    main(sys.argv[1:])
