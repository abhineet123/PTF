from pywinauto import application, mouse
import os
import sys
import time
# from dragonfly import Window

import ctypes
import win32gui
import win32api

from Misc import processArguments

def main():

    params = {
        'win_title': 'PDF-XChange Editor',
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
    # mode = params['mode']
    # wait_t = params['wait_t']
    # scp_dst = params['scp_dst']
    # auth_path = params['auth_path']
    # dst_path = params['dst_path']
    # scp_path = params['scp_path']
    # scp_name = params['scp_name']

    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    GetWindowText = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible

    # Form1.SetFocus()

    # if mode == 0:
    #     data_type = 'filename (from)'
    # elif mode == 1:
    #     data_type = 'filename (to)'
    # elif mode == 2:
    #     data_type = 'log'
    # else:
    #     raise AssertionError('Invalid mode: {}'.format(mode))

    # while True:
    #     k = input('Enter {}\n'.format(data_type))

        # x, y = win32api.GetCursorPos()

        # EnumWindows(EnumWindowsProc(foreach_window), 0)

    titles = []


    def foreach_window(hwnd, lParam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLength(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            titles.append((hwnd, buff.value))
        return True


    win32gui.EnumWindows(foreach_window, None)

    # for i in range(len(titles)):
    #     print(titles[i])

    target_title = [k[1] for k in titles if win_title in k[1]]
    print('target_title: {}'.format(target_title))

    if not target_title:
        print('Window with win_title: {} not found'.format(win_title))
        return

    target_title = target_title[0]
    # print('target_title: {}'.format(target_title))

    try:
        app = application.Application().connect(title=target_title, found_index=0)
    except BaseException as e:
        print('Failed to connect to app for window {}: {}'.format(target_title, e))
        return
    try:
        app_win = app.window(title=target_title)
    except BaseException as e:
        print('Failed to access app window for {}: {}'.format(target_title, e))
        return

    app_win.type_keys("^Q")

    if app_win.exists():
        dialogs = app.windows()
        print(dialogs)

        diag_win = app.window(handle=dialogs[0])
        # print(diag_win)

        diag_win.type_keys("~")


if __name__ == '__main__':
    main()
