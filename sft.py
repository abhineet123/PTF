import time
import win32gui, win32con
import win32api
import pywintypes
import numpy as np
from datetime import datetime

sft_exceptions = ['PotPlayer', 'Free Alarm Clock', 'MPC-HC', 'DisplayFusion',
                  'GPU-Z', 'IrfanView', 'WinRAR', 'Jump List for ']

sft_exceptions_multi = [('XY:(', ') - RGB:(', ', HTML:('), ]


def second_from_top_fn(active_monitor_id, active_win_handle, exit_program,
                       second_from_top, monitors, vwm_win_name, dup_win_names,
                       monitor_id, dup_monitor_ids, duplicate_window,
                       only__maximized, frg_win_handles, frg_monitor_ids,
                       monitor_scale,
                       # global_prev_active_handle,
                       # global_prev_active_name,
                       other_vars=None
                       ):
    # prev_active_win_name = None
    # active_monitor_id = None
    exit_program.value = 0
    vwm_win_names = [vwm_win_name, ] + dup_win_names
    monitor_ids = [monitor_id, ] + dup_monitor_ids
    prev_active_handles = {}
    prev_active_names = {}

    prev_active_called_handles = {}
    prev_active_called_names = {}

    # prev_active_win_exceptions = ['RocketDock', ]

    prev_monitor_id = None
    _global_prev_active_name = None

    centroids = []
    for curr_id, monitor in enumerate(monitors):
        _centroid_x = (int(monitor[0] / monitor_scale) + int(monitor[0] / monitor_scale) + 1920) / 2.0
        _centroid_y = (int(monitor[1] / monitor_scale) + int(monitor[1] / monitor_scale) + 1080) / 2.0

        centroids.append((_centroid_x, _centroid_y))

    # active_names = []
    fixed_xy_bug = 0

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

        # if active_name not in prev_active_win_exceptions:
        #     global_prev_active_handle.value = active_handle

        if not active_name:
            # print('empty active_name')
            continue

        if active_name in vwm_win_names:
            # if fixed_xy_bug:
            #     continue

            # time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
            # print('\n{} :: vwm_win_names'.format(time_stamp))
            # print('frg_win_handles: {}'.format(frg_win_handles))
            # print('active_handle: {}'.format(active_handle))
            # print('active_name: {}'.format(active_name))
            # print('_global_prev_active_name: {}'.format(_global_prev_active_name))

            # print('prev_active_handles: {}'.format(prev_active_handles))
            # print('prev_active_names: {}'.format(prev_active_names))

            # print('prev_active_called_handles: {}'.format(prev_active_called_handles))
            # print('prev_active_called_names: {}'.format(prev_active_called_names))

            if _global_prev_active_name is not None and _monitor_id in prev_active_called_handles:
                prev_active_called_handle = prev_active_called_handles[_monitor_id]
                prev_active_called_name = prev_active_called_names[_monitor_id]

                global_prev_win_handle = win32gui.FindWindow(None, _global_prev_active_name)
                # _prev_active_handle_now = win32gui.FindWindow(None, prev_active_called_name)

                # print('global_prev_win_handle: {}'.format(global_prev_win_handle))
                # print('prev_active_called_name: {}'.format(prev_active_called_name))
                # print('prev_active_called_handle: {}'.format(prev_active_called_handle))

                # if not global_prev_win_handle or (
                # xyplorer_id in _global_prev_active_name and
                # global_prev_win_handle == prev_active_called_handle
                # ):

                # print('XYplorer bug')
                # if not _prev_active_handle_now:
                #     print('_prev_active_handle_now is invalid')

                _win_handle = win32gui.FindWindow(None, vwm_win_name)
                active_win_handle.value = prev_active_called_handle
                win32api.PostMessage(_win_handle, win32con.WM_CHAR, 0x42, 0)

                # fixed_xy_bug = 1
            continue

        # fixed_xy_bug = 0

        # print('active_name: {}'.format(active_name))
        # print('active_names: {}'.format(active_names))

        # active_names.append(active_name)

        _global_prev_active_name = active_name

        # global_prev_active_name.value = active_name

        if any([k in active_name for k in sft_exceptions]):
            # print('sft_exceptions')
            continue

        if any([all([k1 in active_name for k1 in k]) for k in sft_exceptions_multi]):
            # print('sft_exceptions_multi')
            continue

        try:
            tup = win32gui.GetWindowPlacement(active_handle)
        except pywintypes.error as e:
            print('sft :: failed to get window location for {} :: {}'.format(active_name, e))
            continue
        if tup[1] == win32con.SW_SHOWMAXIMIZED:
            # print("sft :: {} is maximized".format(active_name))
            pass
        elif tup[1] == win32con.SW_SHOWMINIMIZED:
            # print("sft :: {} is minimized".format(active_name))
            continue
        elif tup[1] == win32con.SW_SHOWNORMAL:
            if only__maximized:
                # print("sft :: {} is normal".format(active_name))
                continue

        # if active_name and (prev_active_handle is None or prev_active_handle != active_handle) and \
        #         active_name not in [win_name, ] + dup_win_names and \
        #         all([k not in active_name for k in sft_exceptions]) and \
        #         all([any([k1 not in active_name for k1 in k]) for k in sft_exceptions_multi]):

        rect = win32gui.GetWindowRect(active_handle)
        x = (rect[0] + rect[2]) / 2.0
        y = (rect[1] + rect[3]) / 2.0

        dists = [(x - _centroid_x) ** 2 + (y - _centroid_y) ** 2 for (_centroid_x, _centroid_y) in centroids]
        _monitor_id = np.argmin(dists)

        if frg_win_handles:
            if active_handle not in frg_win_handles:
                # print('active_name: {} with handle {} not same as frg_win_handle: {}'.format(
                #     active_name, active_handle, frg_win_handle))
                # print(
                """another win on same monitor was active and now target win is active again
                """
                # )
                prev_active_handles[_monitor_id] = active_handle
                prev_active_names[_monitor_id] = active_name
                continue
            if frg_monitor_ids and _monitor_id not in frg_monitor_ids:
                # print('_monitor_id: {}'.format(_monitor_id))
                # print('frg_monitor_ids: {}'.format(frg_monitor_ids))
                continue
        else:
            if _monitor_id not in monitor_ids:
                # print('_monitor_id: {}'.format(_monitor_id))
                # print('monitor_ids: {}'.format(monitor_ids))
                continue

        try:
            prev_active_handle = prev_active_handles[_monitor_id]
        except KeyError:
            prev_active_handle = None
        else:
            if prev_active_handle is not None and prev_active_handle == active_handle:
                if not frg_win_handles or prev_monitor_id == _monitor_id:
                    # active_name = win32gui.GetWindowText(active_handle)
                    # if ' - XYplorer 20.10' not in active_name:
                    #     print('no XYplorer')
                    continue

        if frg_win_handles and _monitor_id not in prev_active_handles:
            # prev_active_handles[_monitor_id] = []
            # prev_active_names[_monitor_id] = []

            prev_active_called_handles[_monitor_id] = None
            prev_active_called_names[_monitor_id] = None

            prev_active_handles[_monitor_id] = active_handle
            prev_active_names[_monitor_id] = active_name

        # _monitor_id = 0
        # min_dist = np.inf
        # for curr_id, monitor in enumerate(monitors):
        #     _centroid_x = (monitor[0] + monitor[0] + 1920) / 2.0
        #     _centroid_y = (monitor[1] + monitor[1] + 1080) / 2.0
        #     dist = (x - _centroid_x) ** 2 + (y - _centroid_y) ** 2
        #     if dist < min_dist:
        #         min_dist = dist
        #         _monitor_id = curr_id

        # print('active_win_name: {} with pos: {} on monitor {}'.format(active_name, rect, _monitor_id))

        if frg_win_handles or _monitor_id == monitor_id:
            # print('sft: here we are in monitor_id')

            # if prev_active_handle is not None:
            #     prev_active_name = win32gui.GetWindowText(prev_active_handle)
            # else:
            #     prev_active_name = ''
            #
            # if prev_monitor_id == _monitor_id and not win32gui.IsWindow(prev_active_handle):
            #     """
            #     previous active window closed
            #     """
            #     print('sft: previous active window closed: {}'.format(prev_active_name))
            #     prev_active_handles[_monitor_id] = active_handle
            #     prev_monitor_id = _monitor_id
            #     continue
            # else:
            #     print('sft: previous active window still exists: {}'.format(prev_active_name))

            _win_handle = win32gui.FindWindow(None, vwm_win_name)
            # print('sft: _win_handle: {}'.format(_win_handle))

            # active_win_info[0] = _monitor_id
            # active_win_info[1] = active_win_name
            # active_win_info[2] = active_handle

            active_monitor_id.value = _monitor_id
            active_win_handle.value = active_handle
            # active_win_name.value = active_name.encode('utf-8')

            # print('sft: active_monitor_id: {}'.format(active_monitor_id.value))
            # print('sft: active_win_handle: {}'.format(active_win_handle.value))
            # print('sft: active_win_name: {}'.format(active_name))
            #
            # if prev_active_handle is not None:
            #     prev_active_name = win32gui.GetWindowText(prev_active_handle)
            #     print('sft: prev_monitor_id: {}'.format(prev_monitor_id))
            #     print('sft: prev_active_handle: {}'.format(prev_active_handle))
            #     print('sft: prev_active_name: {}'.format(prev_active_name))

            win32api.PostMessage(_win_handle, win32con.WM_CHAR, 0x42, 0)

            # win32gui.ShowWindow(_win_handle, 5)
            # win32gui.SetForegroundWindow(_win_handle)
            #
            # win32gui.ShowWindow(win_handle, 5)
            # win32gui.SetForegroundWindow(win_handle)

            # if other_vars is not None and prev_monitor_id is not None:
            #     active_monitor_id_2, active_win_handle_2, other_win_name = other_vars
            #     print('Sending message to other_win_name: {} with monitor_id: {}, active_handle: {}'.format(
            #         other_win_name, prev_monitor_id, prev_active_handles[prev_monitor_id]))
            #     _win_handle_2 = win32gui.FindWindow(None, other_win_name)
            #     active_monitor_id_2.value = prev_monitor_id
            #     active_win_handle_2.value = prev_active_handles[prev_monitor_id]
            #     win32api.PostMessage(_win_handle_2, win32con.WM_CHAR, 0x42, 0)

            prev_active_handles[_monitor_id] = active_handle
            prev_active_names[_monitor_id] = active_name

            prev_active_called_handles[_monitor_id] = active_handle
            prev_active_called_names[_monitor_id] = active_name

            # prev_active_win_name = active_win_name

        elif duplicate_window:
            # print('sft: here we are in dup_monitor_ids')

            _i = dup_monitor_ids.index(_monitor_id)
            if second_from_top <= _i + 1:
                # print('second_from_top')
                continue

            # if second_from_top > _i + 1:

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

            prev_active_handles[_monitor_id].append(active_handle)
            prev_active_names[_monitor_id].append(active_name)

            prev_active_called_handles[_monitor_id] = active_handle
            prev_active_called_names[_monitor_id] = active_name

            # prev_active_win_name = active_win_name
        # else:
        #     print('_monitor_id: {}'.format(_monitor_id))
        #     print('monitor_ids: {}'.format(monitor_ids))
        #     print('duplicate_window: {}'.format(duplicate_window))

        prev_monitor_id = _monitor_id

    print('{} :: Exiting sft'.format(vwm_win_name))
