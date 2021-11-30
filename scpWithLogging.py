import os
import ctypes
import win32gui
import win32api
from pywinauto import application, mouse

import paramparse

import encrypt_file_aes as encryption


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def run_scp(dst_path, pwd0, scp_dst, scp_path, k, mode, port):
    dst_full_path = '{}/{}'.format(dst_path, k)
    scp_cmd = "pscp -pw {}".format(pwd0)
    if port:
        scp_cmd = '{} -P {}'.format(scp_cmd, port)
    if mode == 0:
        scp_cmd = "{} {}:{}/{} {}".format(scp_cmd, scp_dst, scp_path, k, dst_full_path)
    elif mode == 1:
        scp_cmd = "{} {} {}:{}/".format(scp_cmd, dst_full_path, scp_dst, scp_path)

    # print('Running {}'.format(scp_cmd))
    os.system(scp_cmd)

    if mode == 1:
        rm_cmd = 'rm {}'.format(dst_full_path)
        # print('Running {}'.format(rm_cmd))
        os.system(rm_cmd)


def main():
    params = {
        'win_title': 'The Journal 8',
        'use_ahk': 1,
        'mode': 0,
        'wait_t': 10,
        'scp_dst': '',
        'key_root': '',
        'key_dir': '',
        'auth_root': '',
        'auth_dir': '',
        'auth_file': '',
        'auth_path': '',
        'dst_path': '.',
        'scp_path': '.',
        'scp_name': 'grs',
    }
    paramparse.process_dict(params)

    win_title = params['win_title']
    use_ahk = params['use_ahk']
    mode = params['mode']
    wait_t = params['wait_t']
    scp_dst = params['scp_dst']
    dst_path = params['dst_path']
    scp_path = params['scp_path']
    scp_name = params['scp_name']

    key_root = params['key_root']
    key_dir = params['key_dir']
    auth_root = params['auth_root']
    auth_dir = params['auth_dir']
    auth_file = params['auth_file']

    # Window.get_all_windows()

    auth_path = linux_path(auth_root, auth_dir, auth_file)
    auth_data = open(auth_path, 'r').readlines()
    auth_data = [k.strip() for k in auth_data]

    dst_info = auth_data[0].split(' ')
    name00, name01, ecr0, key0 = dst_info[:4]

    if len(dst_info) > 4:
        port = dst_info[4]

    encryption_params = encryption.Params()
    encryption_params.mode = 1
    encryption_params.root_dir = key_root
    encryption_params.parent_dir = key_dir

    encryption_params.in_file = ecr0
    encryption_params.key_file = key0
    encryption_params.process()
    pwd0 = encryption.run(encryption_params)

    # Form1.SetFocus()
    default_fmy_key = '0'
    if mode == 0:
        data_type = 'filename (from)'
        highlight_key = '2'
    elif mode == 1:
        data_type = 'filename (to)'
        highlight_key = '3'
    elif mode == 2:
        data_type = 'log'
        highlight_key = '4'
    else:
        raise AssertionError('Invalid mode: {}'.format(mode))

    while True:
        k = input('Enter {}\n'.format(data_type))

        x, y = win32api.GetCursorPos()
        # EnumWindows(EnumWindowsProc(foreach_window), 0)
        if use_ahk:
            os.system('paste_with_cat_1')
            run_scp(dst_path, pwd0, scp_dst, scp_path, k, mode, port)
            continue

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

        win32gui.EnumWindows(foreach_window, None)

        # for i in range(len(titles)):
        #     print(titles[i])

        target_title = [k[1] for k in titles if k[1].startswith(win_title)]
        # print('target_title: {}'.format(target_title))

        if not target_title:
            print('Window with win_title: {} not found'.format(win_title))
            run_scp(dst_path, pwd0, scp_dst, scp_path, k, mode, port)
            continue

        target_title = target_title[0]
        # print('target_title: {}'.format(target_title))

        try:
            app = application.Application().connect(title=target_title, found_index=0)
        except BaseException as e:
            print('Failed to connect to app for window {}: {}'.format(target_title, e))
            run_scp(dst_path, pwd0, scp_dst, scp_path, k, mode, port)
            continue
        try:
            app_win = app.window(title=target_title)
        except BaseException as e:
            print('Failed to access app window for {}: {}'.format(target_title, e))
            run_scp(dst_path, pwd0, scp_dst, scp_path, k, mode, port)
            continue

        try:
            # if mode == 2:
            #     enable_highlight = k.strip()
            #     app_win.type_keys("^t~")
            #     app_win.type_keys("^v")
            #     app_win.type_keys("^+a")
            #     if enable_highlight:
            #         app_win.type_keys("^+%a")
            #         # time.sleep(1)
            #         app_win.type_keys("^+z")
            #         app_win.type_keys("{RIGHT}{VK_SPACE}~")
            #     else:
            #         app_win.type_keys("{VK_SPACE}~")
            #
            #     app_win.type_keys("^s")
            #     continue

            app_win.type_keys("^t{VK_SPACE}::{VK_SPACE}1")
            app_win.type_keys("^+a")
            app_win.type_keys("^2")
            # app_win.type_keys("^+1")
            app_win.type_keys("{RIGHT}{LEFT}~")
            app_win.type_keys("^v")
            if mode == 1:
                app_win.type_keys("{LEFT}{RIGHT}{VK_SPACE}to{VK_SPACE}%s" % scp_name)
            # app_win.type_keys("^+a")
            # app_win.type_keys("^{}".format(highlight_key))
            # app_win.type_keys("{LEFT}{RIGHT}~")
            # app_win.type_keys("^{}".format(default_fmy_key))
            app_win.type_keys("~")
            app_win.type_keys("^s")

            mouse.move(coords=(x, y))
        except BaseException as e:
            print('Failed to type entry in app : {}'.format(e))
            pass

        run_scp(dst_path, pwd0, scp_dst, scp_path, k, mode, port)


if __name__ == '__main__':
    main()
