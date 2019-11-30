from datetime import datetime
from multiprocessing import Process
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
        'prob':  0.5,
        'init_sleep': 10,
        'sleep': 60,
    }

    processArguments(sys.argv[1:], params)
    prob = params['prob']
    init_sleep = params['init_sleep']
    sleep = params['sleep']
    script_root = params['script_root']
    script_1 = params['script_1']
    script_2 = params['script_2']

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

    win_name1 = 'VWM_multi_vid_{}'.format(os.path.basename(time_stamp))
    win_name2 = 'VWM_multi_img_{}'.format(os.path.basename(time_stamp))

    # args1 = [k.strip() for k in args1.split(' ') if k.strip()]
    args1 = shlex.split(args1)
    args1 = [k for k in args1[2:] if '%' not in k]
    args1.append('win_name={}'.format(win_name1))
    # args1.append('frg_win_titles={}'.format(frg_win_titles))
    args1.append('other_win_name={}'.format(win_name2))

    # args2 = [k.strip() for k in args2.split(' ') if k.strip()]
    args2 = shlex.split(args2)
    args2 = [k for k in args2[2:] if '%' not in k]
    args2.append('win_name={}'.format(win_name2))
    # args2.append('frg_win_titles={}'.format(frg_win_titles))
    args2.append('other_win_name={}'.format(win_name1))

    vwm1_thread = Process(target=vwm.main, args=(args1,))
    vwm1_thread.start()

    vwm2_thread = Process(target=vwm.main, args=(args2,))
    vwm2_thread.start()

    time.sleep(init_sleep)

    _win_handle_1 = win32gui.FindWindow(None, win_name1)
    _win_handle_2 = win32gui.FindWindow(None, win_name2)

    hidden_win_handle = _win_handle_1
    win32api.PostMessage(hidden_win_handle, win32con.WM_CHAR, 0x68, 0)

    while True:
        time.sleep(sleep)

        num = random.random()

        # print('num: {}'.format(num))

        try:
            if num < prob:
                if hidden_win_handle == _win_handle_1:
                    win32api.PostMessage(_win_handle_2, win32con.WM_CHAR, 0x68, 0)
                    win32api.PostMessage(_win_handle_1, win32con.WM_CHAR, 0x68, 0)
                    hidden_win_handle = _win_handle_2
            else:
                if hidden_win_handle == _win_handle_2:
                    win32api.PostMessage(_win_handle_2, win32con.WM_CHAR, 0x68, 0)
                    win32api.PostMessage(_win_handle_1, win32con.WM_CHAR, 0x68, 0)
                    hidden_win_handle = _win_handle_1
        except:
            break
