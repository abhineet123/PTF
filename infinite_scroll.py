import time
import win32gui, win32api
from pywinauto import application, mouse
import ctypes
import os, sys

from Misc import processArguments


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


def main():
    params = {
        'max_t': 120,
        'relative': 0,
    }
    processArguments(sys.argv[1:], params)

    max_t = params['max_t']

    win_titles = ['Google Chrome']
    try:
        orig_x, orig_y = win32api.GetCursorPos()
        print('GetCursorPos x: {}'.format(orig_x))
        print('GetCursorPos y: {}'.format(orig_y))

        win32gui.EnumWindows(foreach_window, None)

        # for i in range(len(titles)):
        #     print(titles[i])

        target_title = [k[1] for k in titles if all(title in k[1] for title in win_titles)]
        # print('target_title: {}'.format(target_title))

        if not target_title:
            raise IOError('Window with win_titles: {} not found'.format(win_titles))

        target_title = target_title[0]

        target_handle = win32gui.FindWindow(None, target_title)
        rect = win32gui.GetWindowRect(target_handle)

        x = int((rect[0] + rect[2]) / 2)
        y = int((rect[1] + rect[3]) / 2)

        # active_handle = win32gui.GetForegroundWindow()
        # target_title = win32gui.GetWindowText(active_handle)

        print('target_title: {}'.format(target_title))
        print('rect: {}'.format(rect))
        print('x: {}'.format(x))
        print('y: {}'.format(y))

        try:
            app = application.Application().connect(title=target_title, found_index=0)
        except BaseException as e:
            print('Failed to connect to app for window {}: {}'.format(target_title, e))
            exit(0)
        try:
            app_win = app.window(title=target_title)
        except BaseException as e:
            print('Failed to access app window for {}: {}'.format(target_title, e))
            exit(0)

    except BaseException as e:
        print('BaseException: {}'.format(e))
        return

    start_t = time.time()
    while True:
        try:
            app_win.type_keys("{PGDN}")
        except BaseException as e:
            print('BaseException: {}'.format(e))
            break
        end_t = time.time()
        time_elapsed = end_t - start_t
        if time_elapsed > max_t:
            break
        print(time_elapsed)


if __name__ == '__main__':
    main()
