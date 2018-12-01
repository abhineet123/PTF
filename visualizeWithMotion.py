import os
import cv2
import sys, time, random, glob
import numpy as np
from Misc import processArguments, sortKey, stackImages, resizeAR, addBorder
import psutil
import inspect
from datetime import datetime

win_utils_available = 1
try:
    import winUtils

    print('winUtils is available')
except ImportError as e:
    win_utils_available = 0
    print('Failed to import winUtils: {}'.format(e))
try:
    from ctypes import windll, Structure, c_long, byref

    # Get active window id
    # https://msdn.microsoft.com/en-us/library/ms633505
    winID = windll.user32.GetForegroundWindow()
    print("This is your current window ID: {}".format(winID))

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
    print("mouse position x: {} y: {}".format(mousePos.x, mousePos.y))
except ImportError as e:
    mousePos = None

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
    'src_path': '.',
    'width': 1920,
    'height': 1080,
    'min_height_ratio': 0.40,
    'speed': 0.5,
    'show_img': 0,
    'quality': 3,
    'resize': 0,
    'mode': 0,
    'auto_progress': 0,
    'max_switches': 1,
    'transition_interval': 5,
    'random_mode': 0,
    'recursive': 1,
    'fullscreen': 0,
    'reversed_pos': 0,
    'double_click_interval': 0.1,
    'n_images': 1,
    'borderless': 1,
    'set_wallpaper': 0,
    'n_wallpapers': 1000,
    'wallpaper_dir': 'vwm',
}

if __name__ == '__main__':
    p = psutil.Process(os.getpid())
    try:
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except AttributeError:
        os.nice(20)

    processArguments(sys.argv[1:], params)
    src_path = params['src_path']
    _width = width = params['width']
    _height = height = params['height']
    min_height_ratio = params['min_height_ratio']
    speed = params['speed']
    show_img = params['show_img']
    quality = params['quality']
    resize = params['resize']
    mode = params['mode']
    auto_progress = params['auto_progress']
    max_switches = params['max_switches']
    transition_interval = params['transition_interval']
    random_mode = params['random_mode']
    recursive = params['recursive']
    fullscreen = params['fullscreen']
    reversed_pos = params['reversed_pos']
    double_click_interval = params['double_click_interval']
    n_images = params['n_images']
    borderless = params['borderless']
    set_wallpaper = params['set_wallpaper']
    wallpaper_dir = params['wallpaper_dir']
    n_wallpapers = params['n_wallpapers']

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
        print("orig_wp_fname value: {}".format(orig_wp_fname.value))
        # print("orig_wp_fname: {}".format(orig_wp_fname))

        orig_wp_fname = orig_wp_fname.value.decode("utf-8")
        # orig_wp = cv2.imread(orig_wp_fname)

        win_wallpaper_func = ctypes.windll.user32.SystemParametersInfoW

    except BaseException as e:
        print('Wallpaper functionality unavailable: {}'.format(e))
        set_wallpaper = 0

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
    grid_size = None
    try:
        cv_windowed_mode_flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL
    except:
        cv_windowed_mode_flags = cv2.WINDOW_AUTOSIZE

    if mousePos is None:
        curr_monitor = 0
    else:
        curr_monitor = 0
        min_dist = np.inf
        for curr_id, monitor in enumerate(monitors):
            centroid_x = (monitor[0] + monitor[0] + 1920) / 2.0
            centroid_y = (monitor[1] + monitor[1] + 1080) / 2.0
            dist = (mousePos.x - centroid_x) ** 2 + (mousePos.y - centroid_y) ** 2
            if dist < min_dist:
                min_dist = dist
                curr_monitor = curr_id
    print('curr_monitor: ', curr_monitor)

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
    img_fnames = []
    win_offset_x = win_offset_y = 0
    top_border = bottom_border = 0

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
        src_file_list = [item for sublist in src_file_gen for item in sublist]

        # _src_file_list = list(src_file_gen)
        # src_file_list = []
        # for x in _src_file_list:
        #     src_file_list += x
    else:
        src_file_list = [os.path.join(src_dir, k) for k in os.listdir(src_dir) if
                         os.path.splitext(k.lower())[1] in img_exts]

    # src_file_list = [list(x) for x in src_file_list]
    # src_file_list = [x for x in src_file_list]

    # print('src_file_list: ', src_file_list)

    # for (dirpath, dirnames, filenames) in os.walk(src_dir):
    #     print()
    #     print('dirpath', dirpath)
    #     print('filenames', filenames)
    #     print('dirnames', dirnames)
    #     print()

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

    # print('src_file_list: {}'.format(src_file_list))
    # print('img_fname: {}'.format(img_fname))
    # print('img_id: {}'.format(img_id))

    if random_mode:
        print('Random mode enabled')
        src_file_list_rand = list(np.random.permutation(src_file_list))

    src_img_ar, start_row, end_row, start_col, end_col, dst_height, dst_width = [None] * 7
    target_height, target_width, min_height, start_col, end_col, height_ratio = [None] * 6
    dst_img = None

    script_filename = inspect.getframeinfo(inspect.currentframe()).filename
    script_path = os.path.dirname(os.path.abspath(script_filename))

    log_dir = os.path.join(script_path, 'log')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'vwm_log.txt')
    print('Saving log to {}'.format(log_file))

    wallpaper_path = os.path.join(log_dir, wallpaper_dir)
    if not os.path.isdir(wallpaper_path):
        os.makedirs(wallpaper_path)
    print('Saving wallpapers to {}'.format(wallpaper_path))


    def createWindow():
        global mode

        try:
            cv2.destroyWindow(win_name)
        except:
            pass

        if mode == 0:
            if fullscreen:
                cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, 1)
            else:
                cv2.namedWindow(win_name, cv_windowed_mode_flags)
                if win_utils_available:
                    # winUtils.hideBorder(monitors[curr_monitor][0], monitors[curr_monitor][1],
                    #                     width, height, win_name)
                    winUtils.hideBorder2(win_name)
            cv2.moveWindow(win_name, win_offset_x + monitors[curr_monitor][0], win_offset_y + monitors[curr_monitor][1])
        else:
            cv2.namedWindow(win_name, cv_windowed_mode_flags)
            #     winUtils.hideBorder(monitors[2][0], monitors[2][1], width, height, win_name)
            # else:
            if win_utils_available:
                winUtils.hideBorder2(win_name)
                # winUtils.loseFocus(win_name)
            cv2.moveWindow(win_name, win_offset_x + monitors[2][0], win_offset_y + monitors[2][1])

        cv2.setMouseCallback(win_name, mouseHandler)

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
        global mode, height, aspect_ratio
        mode = 1 - mode

        if mode == 0:
            height = int(height / 2.0)
        else:
            height = int(2 * height)

        # print('changeMode :: height: ', height)
        aspect_ratio = float(width) / float(height)
        createWindow()
        loadImage()


    def setGridSize():
        global grid_size, n_images, predef_grid_sizes
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
                n_cols = n_rows = int(np.ceil(np.sqrt(n_images)))
        grid_size = (n_rows, n_cols)


    def loadImage(_type=0, set_grid_size=0):
        global src_img_ar, start_row, end_row, start_col, end_col, dst_height, dst_width, n_switches, img_id, direction
        global target_height, target_width, min_height, start_col, end_col, height_ratio, img_fname, start_time
        global src_start_row, src_start_col, src_end_row, src_end_col, aspect_ratio, \
            src_images, img_fnames, stack_idx, stack_locations, src_img, wp_id, src_file_list_rand, top_border, bottom_border

        if set_grid_size:
            setGridSize()

        if _type != 0:
            top_border = bottom_border = 0
        aspect_ratio = float(width) / float(height)

        if _type != 0 or not src_images:
            if _type == 0:
                img_id -= n_images
            elif _type == -1:
                img_id -= 2 * n_images
            src_images = []
            img_fnames = []
            for _load_id in range(n_images):
                img_id += 1
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

                if img_id >= total_frames:
                    img_id -= total_frames
                    if n_images > 1 and set_wallpaper and random_mode:
                        print('Resetting randomisation')
                        src_file_list_rand = list(np.random.permutation(src_file_list))
                elif img_id < 0:
                    img_id += total_frames

                if random_mode:
                    img_fname = src_file_list_rand[img_id]
                else:
                    img_fname = src_file_list[img_id]

                # src_img_fname = os.path.join(src_dir, img_fname)

                # print('img_id: {}'.format(img_id))
                # print('img_fname: {}'.format(img_fname))

                src_img_fname = img_fname
                src_img = cv2.imread(src_img_fname)
                if src_img is None:
                    raise SystemError('Source image could not be read from: {}'.format(src_img_fname))
                img_fnames.append(img_fname)
                src_images.append(src_img)

        if n_images == 1:
            src_img = src_images[0]
            if top_border > 0:
                src_img = addBorder(src_img, top_border, 'top')
            if bottom_border > 0:
                src_img = addBorder(src_img, bottom_border, 'bottom')
        else:
            src_img, stack_idx, stack_locations = stackImages(src_images, grid_size, borderless=borderless,
                                                              return_idx=1)
            # print('stack_locations: {}'.format(stack_locations))

        if set_wallpaper:
            wp_id = (wp_id + 1) % n_wallpapers
            wp_fname = os.path.join(wallpaper_path, 'wallpaper_{}.jpg'.format(wp_id))
            screensize = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)

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

            if wp_border:
                if n_images == 1 or grid_size[0] % 2 == 1:
                    border_type = 'bottom'
                else:
                    border_type = 'top_and_bottom'
                wp_height_ratio = float(src_img.shape[0]) / float(wp_height)
                src_border = int(wp_border * wp_height_ratio)
                src_img = addBorder(src_img, src_border, border_type)

            src_img_desktop = resizeAR(src_img, wp_width, wp_height)
            src_img = addBorder(src_img, bottom_border, 1)
            wp_end_col = wp_start_col + src_img_desktop.shape[1]
            wp_end_row = wp_start_row + src_img_desktop.shape[0]

            src_img_desktop_full = np.zeros((screensize[1], screensize[0], 3), dtype=np.uint8)
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
            start_row = int((dst_height - src_height) / 2.0)
            start_col = 0
        else:
            dst_height = src_height
            dst_width = int(src_height * aspect_ratio)
            start_col = int((dst_width - src_width) / 2.0)
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
        global speed
        speed += 0.05
        print('speed: ', speed)


    def decreaseSpeed():
        global speed
        speed -= 0.05
        if speed < 0:
            speed = 0
        print('speed: ', speed)


    def setOffsetDiff(dx, dy):
        global row_offset, col_offset

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
        global row_offset, col_offset

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
        global target_height, direction, start_col, start_row, end_row, end_col, n_switches

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
            print('Window minimization unavailable')


    def mouseHandler(event, x, y, flags=None, param=None):
        global img_id, row_offset, col_offset, lc_start_t, rc_start_t, end_exec, fullscreen, \
            direction, target_height, prev_pos, prev_win_pos, speed, old_speed, min_height, min_height_ratio, n_images, src_images
        global win_offset_x, win_offset_y, width, height, top_border, bottom_border
        reset_prev_pos = reset_prev_win_pos = True
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
                # print('flags: {0:b}'.format(flags))
                if flags == 2:
                    loadImage(1)
                elif flags == 10 or flags == 11:
                    direction = -direction
                elif flags == 18:
                    row_offset = col_offset = 0
            elif event == cv2.EVENT_RBUTTONUP:
                pass
            elif event == cv2.EVENT_MBUTTONDOWN:
                flags_str = '{0:b}'.format(flags)
                # print('EVENT_MBUTTONDOWN flags: {:s}'.format(flags_str))
                if flags_str[1] == '1':
                    target_height = min_height
                else:
                    loadImage()
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
                    if flags == 1:
                        loadImage(-1)
                    elif flags == 9:
                        # ctrl
                        target_height = min_height
                    elif flags == 17 or flags == 25:
                        # shift
                        # print('n_images: {}'.format(n_images))
                        if n_images > 1:
                            # print('here we are')
                            resize_ratio = float(dst_img.shape[0]) / float(src_img.shape[0])
                            x_scaled, y_scaled = x / resize_ratio, y / resize_ratio
                            click_found = 0
                            for i in range(n_images):
                                _start_row, _start_col, _end_row, _end_col = stack_locations[i]
                                if x_scaled >= _start_col and x_scaled < _end_col and y_scaled >= _start_row and y_scaled < _end_row:
                                    __idx = stack_idx[i]
                                    fname = '"' + os.path.abspath(img_fnames[__idx]) + '"'
                                    print('Clicked on image {} with id {}:\n {}'.format(i + 1, __idx, fname))
                                    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
                                    open(log_file, 'a').write(time_stamp + "\n" + fname + '\n')
                                    click_found = 1

                                    if flags == 25:
                                        # ctrl + shift
                                        img_id += __idx + 1 - n_images
                                        # print('making img_id: {}'.format(img_id))
                                        n_images = 1
                                        src_images = []
                                        loadImage(0)
                                    break
                            if not click_found:
                                print('x: {}'.format(x))
                                print('y: {}'.format(y))
                                print('resize_ratio: {}'.format(resize_ratio))
                                print('x_scaled: {}'.format(x_scaled))
                                print('y_scaled: {}'.format(y_scaled))
                                print('stack_locations:\n {}\n'.format(stack_locations))
                    # elif flags == 9:
                    #     row_offset = col_offset = 0
                    # setOffset(x, y)
                    # elif flags == 17:
                    #     row_offset = col_offset = 0
                elif event == cv2.EVENT_LBUTTONUP:
                    pass

        except AttributeError as e:
            print('AttributeError: {}'.format(e))
            pass


    win_name = 'VWM'
    createWindow()

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

    img_id += n_images - 1
    loadImage(set_grid_size=1)

    while True:
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
            print('Resizing error: {}'.format(e))
            temp_height, temp_width, _ = temp.shape
            print('temp_height: ', temp_height)
            print('temp_width: ', temp_width)
            if temp_height:
                temp_aspect_ratio = float(temp_width) / float(temp_height)
                print('temp_aspect_ratio: ', temp_aspect_ratio)
            print('_col_offset: ', _col_offset)
            print('_row_offset: ', _row_offset)

        if mode == 0 and not fullscreen:
            temp_height, temp_width, _ = temp.shape
            temp_height_ratio = float(temp_height) / float(height)

            win_start_row = int(max(0, src_start_row - start_row) / temp_height_ratio)
            win_end_row = height - int(max(0, end_row - src_end_row) / temp_height_ratio)

            win_start_col = int(max(0, src_start_col - start_col) / temp_height_ratio)
            win_end_col = width - int(max(0, end_col - src_end_col) / temp_height_ratio)

            dst_img = dst_img[win_start_row:win_end_row, win_start_col:win_end_col, :]

            # print(':: reversed_pos: ', reversed_pos)
            if reversed_pos == 0:
                cv2.moveWindow(win_name, win_offset_x + monitors[curr_monitor][0],
                               win_offset_y + monitors[curr_monitor][1])
            elif reversed_pos == 1:
                cv2.moveWindow(win_name, int(win_offset_x + monitors[curr_monitor][0] + (width - dst_img.shape[1]) / 2),
                               win_offset_y + monitors[curr_monitor][1])
            elif reversed_pos == 2:
                cv2.moveWindow(win_name, win_offset_x + int(monitors[curr_monitor][0] + width - dst_img.shape[1]),
                               win_offset_y + monitors[curr_monitor][1])

            # if win_utils_available:
            #     winUtils.hideBorder2(win_name)

        cv2.imshow(win_name, dst_img)

        # if win_utils_available:
        #     winUtils.loseFocus(win_name)

        # winUtils.hideBorder2(win_name)
        # winUtils.show2(win_name)

        # if win_utils_available:
        #     winUtils.show(win_name, dst_img, 0)
        # else:
        #     cv2.imshow(win_name, dst_img)

        k = cv2.waitKeyEx(1)

        # if k >= 0:
        #     print('k: {}'.format(k))

        if k == 27 or end_exec:
            break
        elif k == 13:
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
        elif k == ord('r'):
            random_mode = 1 - random_mode
            if random_mode:
                print('Random mode enabled')
                src_file_list_rand = list(np.random.permutation(src_file_list))
                img_id = src_file_list_rand.index(img_fname)
            else:
                print('Random mode disabled')
                img_id = src_file_list.index(img_fname)
        elif k == ord('c'):
            auto_progress = 1 - auto_progress
            if auto_progress:
                print('Auto progression enabled')
            else:
                print('Auto progression disabled')
        elif k == ord('q'):
            random_mode = 1 - random_mode
            if random_mode:
                print('Random mode enabled')
                src_file_list_rand = list(np.random.permutation(src_file_list))
            else:
                print('Random mode disabled')
            auto_progress = 1 - auto_progress
            if auto_progress:
                print('Auto progression enabled')
            else:
                print('Auto progression disabled')
        elif k == ord('b'):
            borderless = 1 - borderless
            if borderless:
                print('Borderless stitching enabled')
            else:
                print('Borderless stitching disabled')
            loadImage()
        elif k == ord('n'):
            max_switches -= 1
            if max_switches < 1:
                max_switches = 1
        elif k == ord('N'):
            max_switches += 1
        elif k == ord('1'):
            curr_monitor = 0
            cv2.moveWindow(win_name, win_offset_x + monitors[0][0], win_offset_y + monitors[0][1])
            # createWindow()
        elif k == ord('2'):
            curr_monitor = 1
            cv2.moveWindow(win_name, win_offset_x + monitors[1][0], win_offset_y + monitors[1][1])
            # createWindow()
        elif k == ord('3'):
            curr_monitor = 2
            cv2.moveWindow(win_name, win_offset_x + monitors[2][0], win_offset_y + monitors[2][1])
            # createWindow()
        elif k == ord('4'):
            curr_monitor = 3
            cv2.moveWindow(win_name, win_offset_x + monitors[3][0], win_offset_y + monitors[3][1])
            # createWindow()
        elif k == ord('5'):
            curr_monitor = 4
            cv2.moveWindow(win_name, win_offset_x + monitors[4][0], win_offset_y + monitors[4][1])
            # createWindow()
        elif k == 32:
            is_paused = 1 - is_paused
            if speed == 0:
                speed = old_speed
            else:
                old_speed = speed
                speed = 0
        elif k == ord('p'):
            reversed_pos = (reversed_pos + 1) % 3
            # print('reversed_pos: ', reversed_pos)
            if not reversed_pos:
                cv2.moveWindow(win_name, win_offset_x + monitors[curr_monitor][0],
                               win_offset_y + monitors[curr_monitor][1])
        elif k == ord('t'):
            transition_interval -= 1
            if transition_interval < 0:
                transition_interval = 0
            print('Setting transition interval to: {}'.format(transition_interval))
        elif k == ord('T'):
            transition_interval += 1
            print('Setting transition interval to: {}'.format(transition_interval))
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
            loadImage(1, 1)
        elif k == ord('-'):
            n_images -= 1
            if n_images < 1:
                n_images = 1
            loadImage(1, 1)
        elif k == ord('='):
            predef_n_image_id = (predef_n_image_id + 1) % n_predef_n_images
            n_images = predef_n_images[predef_n_image_id]
            loadImage(1, 1)
        elif k == ord('_'):
            predef_n_image_id -= 1
            if predef_n_image_id < 0:
                predef_n_image_id = n_predef_n_images - 1
            n_images = predef_n_images[predef_n_image_id]
            loadImage(1, 1)
        elif k == ord('!'):
            predef_n_image_id = 0
            n_images = predef_n_images[predef_n_image_id]
            loadImage(1, 1)
        elif k == ord('@'):
            predef_n_image_id = 1
            n_images = predef_n_images[predef_n_image_id]
            loadImage(1, 1)
        elif k == ord('#'):
            predef_n_image_id = 2
            n_images = predef_n_images[predef_n_image_id]
            loadImage(1, 1)
        elif k == ord('$'):
            predef_n_image_id = 3
            n_images = predef_n_images[predef_n_image_id]
            loadImage(1, 1)
        elif k == ord('%'):
            predef_n_image_id = 4
            n_images = predef_n_images[predef_n_image_id]
            loadImage(1, 1)
        elif k == ord('^'):
            predef_n_image_id = 5
            n_images = predef_n_images[predef_n_image_id]
            loadImage(1, 1)
        elif k == ord('&'):
            predef_n_image_id = 6
            n_images = predef_n_images[predef_n_image_id]
            loadImage(1, 1)
        elif k == ord('*'):
            predef_n_image_id = 7
            n_images = predef_n_images[predef_n_image_id]
            loadImage(1, 1)
        elif k == ord('('):
            predef_n_image_id = 8
            n_images = predef_n_images[predef_n_image_id]
            loadImage(1, 1)
        elif k == ord(')'):
            predef_n_image_id = 9
            n_images = predef_n_images[predef_n_image_id]
            loadImage(1, 1)
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
        elif k == 2555904:
            # right
            loadImage(1)
        elif k == 2424832:
            # left
            loadImage(-1)
        elif k == 39 or k == ord('d'):
            loadImage(1)
        elif k == 40 or k == ord('a'):
            loadImage(-1)
        elif k == ord('F') or k == ord('0'):
            if n_images == 1:
                print('"' + os.path.abspath(img_fname) + '"')
            else:
                print()
                for _idx in stack_idx:
                    print('"' + os.path.abspath(img_fnames[_idx]) + '"')
                print()
        elif k == ord('f') or k == ord('/') or k == ord('?'):
            fullscreen = 1 - fullscreen
            createWindow()
            if fullscreen:
                print('fullscreen mode enabled')
            else:
                print('fullscreen mode disabled')

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
            end_time = time.time()
            if end_time - start_time >= transition_interval:
                loadImage(1)

        # print('end_row: ', end_row)
        # print('start_col: ', start_col)
        # print('end_col: ', end_col)

        # print('\n')

    cv2.destroyWindow(win_name)
    if set_wallpaper:
        win_wallpaper_func(SPI_SETDESKWALLPAPER, 0, orig_wp_fname, 0)
