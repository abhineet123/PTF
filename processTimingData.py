from datetime import datetime, timedelta

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk

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


class Params:
    def __init__(self):
        self.cfg = ()
        self.win_titles = ['Timing', 'Google Chrome']
        self.txt_path = 'Z:/Documents/Backup/txtpad'
        self.txt_proc_list = 'processed.log'
        self.recursive = 1
        self.category = 2
        self.ffs = 1
        self.ffs_exe = ''
        self.ffs_root = ''
        self.ffs_ext = ''
        self.ffs_files = []


def foreach_window(hwnd, lParam):
    if IsWindowVisible(hwnd):
        length = GetWindowTextLength(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        GetWindowText(hwnd, buff, length + 1)
        titles.append((hwnd, buff.value))
    return True


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def process(in_txt, verbose=1):
    lines = in_txt.split('\n')
    lines = [line for line in lines if line.strip()]

    if verbose:
        print('lines: {}'.format(lines))

    start_t = None
    curr_t = None
    out_txt = ''
    started = 0
    for line in lines:
        if not started:
            if line.startswith('Timing Data'):
                started = 1
            continue
        if line.startswith('Timing Description: Start Time: '):
            _line = line.replace('Timing Description: Start Time: ', '').strip()
            _line = _line.replace(' a.m.', ' AM')
            _line = _line.replace(' p.m.', ' PM')
            try:
                start_t = datetime.strptime(_line, '%b %d, %Y %I:%M:%S %p')
            except ValueError:
                start_t = datetime.strptime(_line, '%b. %d, %Y %I:%M:%S %p')
            out_txt += start_t.strftime('%d/%m/%Y\t%H:%M:%S') + '\n'
            # print('start_t: {}'.format(start_t))
        if line.startswith('Lap Description: '):
            _line = line.replace('Lap Description: ', '').strip()
            if ' <Stopped>' in _line:
                _line = _line.replace(' <Stopped>', '')
            out_txt += _line.replace('...', '\t')
        elif line.startswith('Lap Time: '):
            _line = line.replace('Lap Time: ', '').strip()
            _line_data = _line.split('.')
            # print('_line_data: {}'.format(_line_data))

            _line = '{}:{}'.format(_line_data[0], int(_line_data[1]) * 10000)
            # print('_line: {}'.format(_line))

            lap_t = datetime.strptime(_line, '%H:%M:%S:%f')
            # print('curr_t: {}'.format(lap_t))
            out_txt += lap_t.strftime('%H:%M:%S') + '.{}\t'.format(_line_data[1])
        elif line.startswith('Lap Total Time: '):
            _line = line.replace('Lap Total Time: ', '').strip()
            _line_data = _line.split('.')
            # print('_line_data: {}'.format(_line_data))

            _line = '{}:{}'.format(_line_data[0], int(_line_data[1]) * 10000)
            # print('_line: {}'.format(_line))

            lap_total_t = datetime.strptime(_line, '%H:%M:%S:%f')
            # print('curr_t: {}'.format(lap_t))
            out_txt += lap_total_t.strftime('%H:%M:%S') + '.{}\t'.format(_line_data[1])
            curr_t = start_t + timedelta(hours=lap_total_t.hour, minutes=lap_total_t.minute,
                                         seconds=lap_total_t.second, microseconds=lap_total_t.microsecond)
            out_txt += curr_t.strftime('%H:%M:%S') + '.{}\n'.format(int(curr_t.microsecond / 10000))
    return out_txt


def copy_to_clipboard(out_txt):
    try:
        import pyperclip

        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


def main():
    params = Params()
    paramparse.process(params)

    try:
        in_txt = Tk().clipboard_get()
    except BaseException as e:
        print('Tk().clipboard_get() failed: {}'.format(e))
        lines = []
    else:
        lines = in_txt.split('\n')
        lines = [line for line in lines if line.strip()]

    out_txt = ''

    if lines:
        if all(line.endswith('.ogg') for line in lines):
            out_lines = [line.split('-')[1].split('.')[0].replace('_', ':') for line in lines]
            if params.category > 0:
                out_lines = ['{} :: {}'.format(line, params.category) for line in out_lines]

            out_txt = '\n'.join(sorted(out_lines))

        else:
            try:
                out_txt = process(in_txt)
            except:
                out_txt = ''

    if not out_txt:

        if params.ffs:
            for _ffs_file in params.ffs_files:
                ffs_path = os.path.join(params.ffs_root, _ffs_file + '.' + params.ffs_ext)
                ffs_cmd = '{} "{}"'.format(params.ffs_exe, ffs_path)
                print(ffs_cmd)
                os.system(ffs_cmd)

        if params.txt_path:

            assert os.path.isdir(params.txt_path), "embedded text path: {}".format(params.txt_path)

            if params.recursive:
                files_gen = [[linux_path(dirpath, f) for f in filenames if
                              f.endswith('.txt') and f.startswith('Timing')]
                             for (dirpath, dirnames, filenames) in os.walk(params.txt_path, followlinks=True)]
                files = [item for sublist in files_gen for item in sublist]
            else:
                files = os.listdir(params.txt_path)
                files = [linux_path(params.txt_path, k) for k in files if k and k.endswith('.txt')]

            txt_proc_list_path = linux_path(params.txt_path, params.txt_proc_list)

            if os.path.isfile(txt_proc_list_path):
                processed_files = open(txt_proc_list_path, 'r').readlines()
                processed_files = [k.strip().split('\t')[1] for k in processed_files if k.strip()]

                files = [k for k in files if k not in processed_files]

            files.sort(key=os.path.getmtime)
            n_files = len(files)

            if n_files > 1:
                print('found {} new files:\n{}'.format(n_files, '\n'.join(files)))

            for file_id, file in enumerate(files):
                print('reading file {} / {}: {}'.format(file_id + 1, n_files, file))

                # file = dst_file

                in_txt = open(file, 'r').read()
                out_txt = process(in_txt, verbose=0)
                print(out_txt)

                copy_to_clipboard(out_txt)

                with open(txt_proc_list_path, 'r+') as f:
                    content = f.read()
                    f.seek(0, 0)

                    timestamp_str = datetime.now().strftime("%y%m%d %H:%M:%S.%f")[:-4]

                    txt = '{}\t{}\n'.format(timestamp_str, file)
                    f.write(txt + content)

                # dst_file = file.replace('.txt', '.log')
                # shutil.move(file, dst_file)
                os.system("vscode {}".format(file))

                _ = input('\npress any key to continue\n')

            return

        # time.sleep(1)
        try:
            orig_x, orig_y = win32api.GetCursorPos()
            print('GetCursorPos x: {}'.format(orig_x))
            print('GetCursorPos y: {}'.format(orig_y))

            win32gui.EnumWindows(foreach_window, None)

            # for i in range(len(titles)):
            #     print(titles[i])

            target_title = [k[1] for k in titles if all(title in k[1] for title in params.win_titles)]
            # print('target_title: {}'.format(target_title))

            if not target_title:
                raise IOError('Window with win_titles: {} not found'.format(params.win_titles))

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
            app_win.type_keys("^a")
            app_win.type_keys("^c")

            mouse.move(coords=(x, y))

            mouse.click(button='left', coords=(x, y))
            mouse.click(button='left', coords=(x, y))

            mouse.move(coords=(orig_x, orig_y))

        except BaseException as e:
            print('BaseException: {}'.format(e))

        in_txt = Tk().clipboard_get()
        out_txt = process(in_txt)

    print(out_txt)
    # with open(out_fname, 'w') as out_fid:
    #     out_fid.write(out_txt)

    copy_to_clipboard(out_txt)


if __name__ == '__main__':
    main()
