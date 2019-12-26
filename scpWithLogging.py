from pywinauto import application, mouse
import os
import sys
import time
# from dragonfly import Window

import ctypes
import win32gui
import win32api

from Misc import processArguments

if __name__ == '__main__':
    params = {
        'win_title': 'The Journal 8',
        'mode': 0,
        'wait_t': 10,
        'scp_dst': '',
        'auth_path': '',
        'dst_path': '.',
        'scp_path': '.',
        'scp_name': 'grs',
    }
    processArguments(sys.argv[1:], params)
    win_title = params['win_title']
    mode = params['mode']
    wait_t = params['wait_t']
    scp_dst = params['scp_dst']
    auth_path = params['auth_path']
    dst_path = params['dst_path']
    scp_path = params['scp_path']
    scp_name = params['scp_name']

    # Window.get_all_windows()
    auth_data = open(auth_path, 'r').readlines()
    auth_data = [k.strip() for k in auth_data]

    name00, name01, pwd0 = auth_data[0].split(' ')

    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    GetWindowText = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible

    titles = []


    def foreach_window(hwnd, lParam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLength(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            titles.append((hwnd, buff.value))
        return True


    # EnumWindows(EnumWindowsProc(foreach_window), 0)

    win32gui.EnumWindows(foreach_window, None)

    # for i in range(len(titles)):
    #     print(titles[i])

    target_title = [k[1] for k in titles if k[1].startswith(win_title)]
    # print('target_title: {}'.format(target_title))

    if not target_title:
        raise IOError('Window with win_title: {} not found'.format(win_title))

    target_title = target_title[0]
    # print('target_title: {}'.format(target_title))

    app = application.Application().connect(title=target_title)
    app_win = app.window(title=target_title)
    # Form1.SetFocus()

    if mode == 2:
        data_type = 'log'
    else:
        data_type = 'filename'

    while True:
        k = input('Enter {}\n'.format(data_type))

        x, y = win32api.GetCursorPos()

        if mode == 2:
            enable_highlight = k.strip()
            app_win.type_keys("^t~")
            app_win.type_keys("^v")
            if enable_highlight:
                app_win.type_keys("^+%a")
                # time.sleep(1)
                app_win.type_keys("^+z")
                app_win.type_keys("{RIGHT}{VK_SPACE}~")
            else:
                app_win.type_keys("{VK_SPACE}~")

            app_win.type_keys("^s")
            continue

        app_win.type_keys("^t~")
        app_win.type_keys("^b")
        app_win.type_keys("^v")
        app_win.type_keys("^b")
        if mode == 1:
            app_win.type_keys("{RIGHT}{VK_SPACE}to{VK_SPACE}%s" % scp_name)
        app_win.type_keys("~")
        app_win.type_keys("^s")

        mouse.move(coords=(x, y))

        dst_full_path = '{}/{}'.format(dst_path, k)
        if mode == 0:
            scp_cmd = "pscp -pw {} {}:{}/{} {}".format(pwd0, scp_dst, scp_path, k, dst_full_path)
        elif mode == 1:
            scp_cmd = "pscp -pw {} {} {}:{}/".format(pwd0, dst_full_path, scp_dst, scp_path)

        # print('Running {}'.format(scp_cmd))
        os.system(scp_cmd)

        if mode == 1:
            rm_cmd = 'rm {}'.format(dst_full_path)
            # print('Running {}'.format(rm_cmd))
            os.system(rm_cmd)


