import time
import win32gui, win32con
import win32api
import numpy as np


sft_exceptions = ['PotPlayer', 'Free Alarm Clock', 'MPC-HC', 'DisplayFusion',
                  'GPU-Z', 'IrfanView', 'WinRAR', 'Jump List for ']

sft_exceptions_multi = [('XY:(', ') - RGB:(', ', HTML:('), ]


def second_from_top_fn(active_monitor_id, active_win_handle, exit_program,
                       second_from_top, monitors, win_name, dup_win_names,
                       monitor_id, dup_monitor_ids, duplicate_window):
    prev_active_handle = None
    # prev_active_win_name = None
    # active_monitor_id = None
    exit_program.value = 0
    while True:
        _exit_program = int(exit_program.value)
        if _exit_program:
            break
        time.sleep(1)
        # print('button: {}'.format(button))
        # print('pressed: {}'.format(pressed))

        active_handle = win32gui.GetForegroundWindow()
        active_name = win32gui.GetWindowText(active_handle)
        # print('active_name: {}'.format(active_name))

        if active_name and (prev_active_handle is None or prev_active_handle != active_handle) and \
                active_name not in [win_name, ] + dup_win_names and \
                all([k not in active_name for k in sft_exceptions]) and \
                all([any([k1 not in active_name for k1 in k]) for k in sft_exceptions_multi]):

            tup = win32gui.GetWindowPlacement(active_handle)
            if tup[1] == win32con.SW_SHOWMAXIMIZED:
                # print("sft :: {} is maximized".format(active_name))
                pass
            elif tup[1] == win32con.SW_SHOWMINIMIZED:
                # print("sft :: {} is minimized".format(active_name))
                continue
            elif tup[1] == win32con.SW_SHOWNORMAL:
                # print("sft :: {} is normal".format(active_name))
                continue

            prev_active_handle = active_handle
            # prev_active_win_name = active_win_name

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

                # active_win_info[0] = _monitor_id
                # active_win_info[1] = active_win_name
                # active_win_info[2] = active_handle

                active_monitor_id.value = _monitor_id
                active_win_handle.value = active_handle
                # active_win_name.value = active_name.encode('utf-8')

                # print('sft: active_monitor_id: {}'.format(active_monitor_id))
                # print('sft: active_win_handle: {}'.format(active_win_handle))
                # print('sft: active_win_name: {}'.format(active_win_name))

                win32api.PostMessage(_win_handle, win32con.WM_CHAR, 0x42, 0)

                # win32gui.ShowWindow(_win_handle, 5)
                # win32gui.SetForegroundWindow(_win_handle)
                #
                # win32gui.ShowWindow(win_handle, 5)
                # win32gui.SetForegroundWindow(win_handle)

            elif duplicate_window and _monitor_id in dup_monitor_ids:
                _i = dup_monitor_ids.index(_monitor_id)
                if second_from_top > _i + 1:


                    _win_handle = win32gui.FindWindow(None, dup_win_names[_i])

                    # active_win_info[0] = _monitor_id
                    # active_win_info[1] = active_win_name
                    # active_win_info[2] = active_handle

                    active_monitor_id.value = _monitor_id
                    active_win_handle.value = active_handle
                    # active_win_name.value = active_name.encode('utf-8')

                    # print('sft: active_monitor_id: {}'.format(active_monitor_id))
                    # print('sft: active_win_handle: {}'.format(active_win_handle))
                    # print('sft: active_win_name: {}'.format(active_win_name))

                    win32api.PostMessage(_win_handle, win32con.WM_CHAR, 0x44, 0)
                    # print('temp: {}'.format(temp))

                    # win32gui.ShowWindow(_win_handle, 5)
                    # win32gui.SetForegroundWindow(_win_handle)
                    #
                    # win32gui.ShowWindow(win_handle, 5)
                    # win32gui.SetForegroundWindow(win_handle)

    print('Exiting sft')
