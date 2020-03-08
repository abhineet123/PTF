from datetime import datetime, timedelta

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk

import time
import win32gui, win32api
from pywinauto import application, mouse

try:
    in_txt = Tk().clipboard_get()
except:
    lines = []
else:
    lines = in_txt.split('\n')
    lines = [line for line in lines if line.strip()]

if all(line.endswith('.ogg') for line in lines):
    out_lines = [line.split('-')[1].split('.')[0].replace('_', ':') for line in lines]
    out_txt = '\n'.join(sorted(out_lines))
else:

    time.sleep(1)

    try:
        active_handle = win32gui.GetForegroundWindow()
        target_title = win32gui.GetWindowText(active_handle)

        print('target_title: {}'.format(target_title))

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

        x, y = win32api.GetCursorPos()

        mouse.click(button='left', coords=(x, y))
        mouse.click(button='left', coords=(x, y))
    except BaseException as e:
        print('BaseException: {}'.format(e))

    in_txt = Tk().clipboard_get()

    lines = in_txt.split('\n')
    lines = [line for line in lines if line.strip()]

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

print(out_txt)
# with open(out_fname, 'w') as out_fid:
#     out_fid.write(out_txt)
try:
    import pyperclip

    pyperclip.copy(out_txt)
    spam = pyperclip.paste()
except BaseException as e:
    print('Copying to clipboard failed: {}'.format(e))
