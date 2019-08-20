from threading import Event
import os
import psutil
import inspect
import mouse
import sys
import win32gui, win32con, win32com.client
import numpy as np

from Misc import processArguments

params = {
    'win_name': '.',
    'win_name2': '',
    'check_interval': 1000,
    'win_names_fname': 'vwm_win_names.txt',
}
if __name__ == '__main__':
    p = psutil.Process(os.getpid())
    try:
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except AttributeError:
        os.nice(20)

    processArguments(sys.argv[1:], params)
    win_name = params['win_name']
    win_name2 = params['win_name2']
    check_interval = params['check_interval']
    win_names_fname = params['win_names_fname']

    if not win_name or not win_name2:
        script_filename = inspect.getframeinfo(inspect.currentframe()).filename
        script_path = os.path.dirname(os.path.abspath(script_filename))

        log_dir = os.path.join(script_path, 'log')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        win_names_file = os.path.join(log_dir, win_names_fname)
        print('Reading vwm win_names from {}'.format(win_names_file))

        lines = open(win_names_file, 'r').readlines()
        win_name, win_name2 = [line.strip() for line in lines if line.strip()]

    print('win_name: {}'.format(win_name))
    print('win_name2: {}'.format(win_name2))

    monitors = [
        [0, 0],
        [-1920, 0],
        [0, -1080],
        [1920, 0],
        [1920, -1080],
    ]

    prev_active_win_name = None

    interrupt_wait = Event()

    def getMonitorID(_win_name):
        _win_handle = win32gui.FindWindow(None, _win_name)
        _rect = win32gui.GetWindowRect(_win_handle)
        x = (_rect[0] + _rect[2]) / 2.0
        y = (_rect[1] + _rect[3]) / 2.0

        _monitor_id = 0
        min_dist = np.inf
        for curr_id, monitor in enumerate(monitors):
            _centroid_x = (monitor[0] + monitor[0] + 1920) / 2.0
            _centroid_y = (monitor[1] + monitor[1] + 1080) / 2.0
            dist = (x - _centroid_x) ** 2 + (y - _centroid_y) ** 2
            if dist < min_dist:
                min_dist = dist
                _monitor_id = curr_id

        return _monitor_id, _win_handle, _rect


    def mouse_click_callback():
        global prev_active_win_name

        interrupt_wait.set()

    mouse.on_click(mouse_click_callback, args=())

    while True:
        interrupt_wait.wait(check_interval)
        interrupt_wait.clear()

        active_win_name = win32gui.GetWindowText(win32gui.GetForegroundWindow())
        print('active_win_name: {}'.format(active_win_name))
        # (prev_active_win_name is None or prev_active_win_name != active_win_name)
        if active_win_name not in (
                win_name, win_name2):
            prev_active_win_name = active_win_name

            _monitor_id, _win_handle, rect =  getMonitorID(active_win_name)

            monitor_id, win_handle, _ = getMonitorID(win_name)
            monitor_id2, win_handle2, _ = getMonitorID(win_name2)

            print('_monitor_id: {}'.format(_monitor_id))
            print('monitor_id: {}'.format(monitor_id))
            print('monitor_id2: {}'.format(monitor_id2))

            print('_win_handle: {}'.format(_win_handle))
            print('win_handle: {}'.format(win_handle))
            print('win_handle2: {}'.format(win_handle2))

            print('active_win_name: {} with pos: {} on monitor {}'.format(active_win_name, rect, _monitor_id))

            if _monitor_id == monitor_id:
                # shell = win32com.client.Dispatch("WScript.Shell")
                # shell.SendKeys('%')
                win32gui.ShowWindow(win_handle, 5)
                win32gui.SetForegroundWindow(win_handle)

                win32gui.ShowWindow(_win_handle, 5)
                win32gui.SetForegroundWindow(_win_handle)
            elif _monitor_id == monitor_id2:
                # shell = win32com.client.Dispatch("WScript.Shell")
                # shell.SendKeys('%')
                win32gui.ShowWindow(win_handle2, 5)
                win32gui.SetForegroundWindow(win_handle2)

                win32gui.ShowWindow(_win_handle, 5)
                win32gui.SetForegroundWindow(_win_handle)