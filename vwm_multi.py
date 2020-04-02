from datetime import datetime
from multiprocessing import Process
import multiprocessing
import sys
import os
import time
import random
import shlex
import win32api
import win32gui, win32con

import visualizeWithMotion as vwm
from Misc import processArguments

if __name__ == '__main__':

    params = {
        'script_root': 'scripts',
        'script_1': 'vw32ntjv.cmd',
        'script_2': 'vw32ntj.cmd',
        'prob': 0.5,
        'init_sleep': 5,
        'sleep': 0,
        'sleep_1': 60,
        'sleep_2': 60,
        'start_sleep': 60,
    }

    processArguments(sys.argv[1:], params)
    prob = params['prob']
    init_sleep = params['init_sleep']
    sleep = params['sleep']
    sleep_1 = params['sleep_1']
    sleep_2 = params['sleep_2']
    start_sleep = params['start_sleep']
    script_root = params['script_root']
    script_1 = params['script_1']
    script_2 = params['script_2']

    if sleep > 0:
        if sleep_1 <= 0:
            sleep_1 = sleep
        if sleep_2 <= 0:
            sleep_2 = sleep
        if start_sleep <= 0:
            start_sleep = sleep

    print('sleep_1: {}'.format(sleep_1))
    print('sleep_2: {}'.format(sleep_2))
    print('start_sleep: {}'.format(start_sleep))

    curr_path = os.path.dirname(os.path.abspath(__file__))

    script_1_path = os.path.join(curr_path, script_root, script_1)
    script_2_path = os.path.join(curr_path, script_root, script_2)

    args1_lines = open(script_1_path, 'r').readlines()
    args1 = [k for k in args1_lines if k.startswith('python3 ')]
    assert len(args1) == 1, "Invalid script: {} with contents: {}".format(script_1_path, args1_lines)
    args1 = args1[0]

    args2_lines = open(script_2_path, 'r').readlines()
    args2 = [k for k in args2_lines if k.startswith('python3 ')]
    assert len(args2) == 1, "Invalid script: {} with contents: {}".format(script_2_path, args2_lines)
    args2 = args2[0]

    # frg_win_titles = 'The Journal 8,f,t,XYplorer 20.10'

    # args1 = 'on_top=0 top_border=0 keep_borders=1 n_images=1 random_mode=1 auto_progress=1 fps=15 monitor_id=0 ' \
    #         'duplicate_window=0 second_from_top=1 win_offset_y=0 width=1920 height=1050 ' \
    #         'src_dirs=vids/20/2/14,vids/20/2/13 ' \
    #         'only_maximized=0 ' \
    #         'reversed_pos=2 video_mode=2 multi_mode=1 auto_progress_video=1 preserve_order=1 lazy_video_load=0'
    # args2 = 'on_top=0 top_border=0 keep_borders=1 n_images=2 ' \
    #         'random_mode=1 auto_progress=1 transition_interval=30 monitor_id=0  ' \
    #         'duplicate_window=0 second_from_top=1 win_offset_y=0 width=1920 height=1050 ' \
    #         'src_dirs=vids/20/2/13**10,vids/20/2_patches**10,!20/9,!20/10,20,20/2*5,20/1*3,20/1/1_0*5,20/1/1_1*25,' \
    #         '20/1/1_8*8,20/1/1_9*20,20/1/1_3*4,20/1/1_5*15,20/1/1_2*15,20/1/1_4*20 ' \
    #         'only_maximized=0 reversed_pos=2'

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    win_1 = 'vid_{}'.format(os.path.basename(time_stamp))
    win_2 = 'img_{}'.format(os.path.basename(time_stamp))

    # args1 = [k.strip() for k in args1.split(' ') if k.strip()]
    args1 = shlex.split(args1)
    args1 = [k for k in args1[2:] if '%' not in k]
    args1.append('win_name={}'.format(win_1))
    # args1.append('frg_win_titles={}'.format(frg_win_titles))
    args1.append('other_win_name={}'.format(win_2))
    args1.append('log_color={}'.format('cyan'))

    # args2 = [k.strip() for k in args2.split(' ') if k.strip()]
    args2 = shlex.split(args2)
    args2 = [k for k in args2[2:] if '%' not in k]
    args2.append('win_name={}'.format(win_2))
    # args2.append('frg_win_titles={}'.format(frg_win_titles))
    args2.append('other_win_name={}'.format(win_1))
    args2.append('log_color={}'.format('green'))

    exit_program = multiprocessing.Value('L', 0, lock=False)

    # sft_active_monitor_id_1 = multiprocessing.Value('I', lock=False)
    # sft_active_win_handle_1 = multiprocessing.Value('L', lock=False)
    #
    # sft_active_monitor_id_2 = multiprocessing.Value('I', lock=False)
    # sft_active_win_handle_2 = multiprocessing.Value('L', lock=False)
    #
    # sft_vars_1 = (sft_active_monitor_id_1, sft_active_win_handle_1, sft_active_monitor_id_2, sft_active_win_handle_2)
    # sft_vars_2 = (sft_active_monitor_id_2, sft_active_win_handle_2, sft_active_monitor_id_1, sft_active_win_handle_1)

    thread_1 = Process(target=vwm.main,
                       args=(args1, exit_program,
                             # sft_vars_1
                             ))
    thread_1.start()

    thread_2 = Process(target=vwm.main,
                       args=(args2, exit_program,
                             # sft_vars_2
                             ))
    thread_2.start()

    time.sleep(init_sleep)
    handle_2 = handle_1 = 0

    while not handle_1:
        handle_1 = win32gui.FindWindow(None, win_1)
        if handle_1:
            break
        print('Waiting for {} handle'.format(win_1))
        time.sleep(2)

    while not handle_2:
        handle_2 = win32gui.FindWindow(None, win_2)
        if handle_2:
            break
        print('Waiting for {} handle'.format(win_2))
        time.sleep(2)

    print('handle_1: {}'.format(handle_1))
    print('handle_2: {}'.format(handle_2))

    if not handle_1 or not handle_2:
        raise AssertionError('Invalid handle found')

    hidden_win_handle = handle_1
    win32api.PostMessage(hidden_win_handle, win32con.WM_CHAR, 0x68, 0)
    _sleep = start_sleep

    switch_t = time.time()

    # prev_shown_time = {
    #     win_1: switch_t,
    #     win_2: switch_t,
    # }
    # _prev_shown_time = {
    #     win_1: switch_t,
    #     win_2: switch_t,
    # }
    visible_duration = {
        win_1: 0,
        win_2: 0,
    }
    _visible_time = {
        win_1: 0,
        win_2: 0,
    }
    visible_ratio = {
        win_1: 0,
        win_2: 0,
    }

    # print('sleep_1: {}'.format(sleep_1))
    # print('sleep_2: {}'.format(sleep_2))
    # print('start_sleep: {}'.format(start_sleep))
    # print('_sleep: {}'.format(_sleep))

    while True:
        time.sleep(_sleep)

        _exit_program = int(exit_program.value)
        if _exit_program:
            break

        num = random.random()
        # print('num: {}'.format(num))

        switch_t = time.time()
        try:
            if num < prob:
                if hidden_win_handle == handle_1:
                    """hide win_2 / show win_1"""
                    win32api.PostMessage(handle_2, win32con.WM_CHAR, 0x68, 0)
                    win32api.PostMessage(handle_1, win32con.WM_CHAR, 0x68, 0)
                    hidden_win_handle = handle_2

                    # _visible_time = switch_t - _prev_shown_time[win_2]
                    # visible_duration[win_2] += switch_t - prev_shown_time[win_2]
                    # prev_shown_time[win_1] = switch_t
                    # _prev_shown_time[win_1] = switch_t

                    _visible_time[win_2] += sleep_2
                    visible_duration[win_2] += sleep_2
                    _sleep = sleep_1
                    print('\nHiding {} after being visible for {:.2f}\n'.format(win_2, _visible_time[win_2]))
                    _visible_time[win_2] = 0
                else:
                    """win_1 remains visible"""
                    # visible_duration[win_1] += switch_t - prev_shown_time[win_1]
                    # prev_shown_time[win_1] = switch_t

                    visible_duration[win_1] += sleep_1
                    _visible_time[win_1] += sleep_1
            else:
                if hidden_win_handle == handle_2:
                    """hide win_1 / show win_2"""
                    win32api.PostMessage(handle_2, win32con.WM_CHAR, 0x68, 0)
                    win32api.PostMessage(handle_1, win32con.WM_CHAR, 0x68, 0)
                    hidden_win_handle = handle_1

                    # _visible_time = switch_t - _prev_shown_time[win_1]
                    # visible_duration[win_1] += switch_t - prev_shown_time[win_1]
                    # prev_shown_time[win_2] = switch_t
                    # _prev_shown_time[win_2] = switch_t

                    visible_duration[win_1] += sleep_1
                    _visible_time[win_1] += sleep_1

                    _sleep = sleep_2
                    print('\nHiding {} after being visible for {:.2f}\n'.format(win_1, _visible_time[win_1]))
                    _visible_time[win_1] = 0

                else:
                    """win_2 remains visible"""
                    # visible_duration[win_2] += switch_t - prev_shown_time[win_2]
                    # prev_shown_time[win_2] = switch_t

                    visible_duration[win_2] += sleep_2
                    _visible_time[win_2] += sleep_2

            total_duration = visible_duration[win_1] + visible_duration[win_2]
            visible_ratio[win_1] = visible_duration[win_1] / total_duration
            visible_ratio[win_2] = visible_duration[win_2] / total_duration
            print('\n'.join('{} : {:.2f} {:.2f}%%'.format(k, visible_duration[k], visible_ratio[k] * 100)
                            for k in (win_1, win_2)) + '\n')
        except BaseException as e:
            print('Exiting on exception: {}'.format(e))
            break
