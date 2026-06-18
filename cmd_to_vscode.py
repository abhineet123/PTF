from datetime import datetime, timedelta

import time
import win32gui, win32api
from pywinauto import application, mouse
import os
import shutil
import ctypes

import subprocess
import paramparse

EnumWindows = ctypes.windll.user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
GetWindowText = ctypes.windll.user32.GetWindowTextW
GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
IsWindowVisible = ctypes.windll.user32.IsWindowVisible

titles = []


class Params(paramparse.CFG):
    def __init__(self):
        paramparse.CFG.__init__(self)

        self.cmd = Params.CMD()

    class CMD:
        def __init__(self):
            self.type = ''
            self.type_3 = ''
            self.paste = ''
            self.paste_3 = ''
            self.task = ''
            self.task_time = ''


def copy_from_clipboard():
    try:
        import win32clipboard

        win32clipboard.OpenClipboard()
        in_txt = win32clipboard.GetClipboardData()
    except BaseException as e:
        print('GetClipboardData failed: {}'.format(e))
        win32clipboard.CloseClipboard()
        return None
    win32clipboard.CloseClipboard()
    return in_txt


def copy_to_clipboard(out_txt, print_txt=0):
    if print_txt:
        print(out_txt)
    try:
        # win32clipboard.OpenClipboard()
        # win32clipboard.SetClipboardText(out_txt)
        import pyperclip

        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))
        return
    time.sleep(0.1)
    # win32clipboard.CloseClipboard()


def main():
    params = Params()
    paramparse.process(params)

    in_txt = copy_from_clipboard()

    in_txt=in_txt.rstrip()

    out_txt = f'			"args": "{in_txt}",\n'

    copy_to_clipboard(out_txt)



if __name__ == '__main__':
    main()
