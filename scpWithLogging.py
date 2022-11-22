import os
# import inspect
import shutil
import ctypes
import win32gui
import win32api
from pywinauto import application, mouse
from datetime import datetime

import paramparse

import encrypt_file_aes as encryption


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def run_scp(dst_path, pwd0, scp_dst, scp_path, file_to_transfer, mode, port, log_file=''):
    # print('dst_path: {}'.format(dst_path))
    # print('pwd0: {}'.format(pwd0))
    # print('scp_dst: {}'.format(scp_dst))
    # print('scp_path: {}'.format(scp_path))
    # print('k: {}'.format(k))
    # print('mode: {}'.format(mode))
    # print('port: {}'.format(port))

    dst_full_path = linux_path(dst_path, file_to_transfer)

    scp_cmd = "scp -r -i {}".format(pwd0)

    # if mode == -1 or mode == -2:
    #     scp_cmd = "scp -i {}".format(pwd0)
    # else:
    #     scp_cmd = "pscp -pw {}".format(pwd0)

    if port:
        scp_cmd = '{} -P {}'.format(scp_cmd, port)

    # if mode == 0:
    #     scp_cmd = "{} {}:{}/{} {}".format(scp_cmd, scp_dst, scp_path, k, dst_full_path)
    # if mode == 1:
    #     scp_cmd = "{} {} {}:{}/".format(scp_cmd, dst_full_path, scp_dst, scp_path)
    scp_full_path = linux_path(scp_path, file_to_transfer)

    invalid_chars = [' ', ')', '(', '&']
    for invalid_char in invalid_chars:
        scp_full_path = scp_full_path.replace(invalid_char, f'\\{invalid_char}')
        dst_full_path = dst_full_path.replace(invalid_char, f'\\{invalid_char}')

    # scp_full_path = scp_full_path.replace(' ', '\ ').replace(')', '\)').replace('(', '\(')
    # dst_full_path = dst_full_path.replace(' ', '\ ').replace(')', '\)').replace('(', '\(')

    if mode == 0 or mode == -1:
        scp_cmd = f'{scp_cmd} {scp_dst}:"{scp_full_path}" "{dst_path}"'
    elif mode == 1 or mode == -2:
        scp_cmd = f'{scp_cmd} "{dst_full_path}" "{scp_dst}:{scp_full_path}"'

    print('Running {}'.format(scp_cmd))
    os.system(scp_cmd)

    dst_path_full = linux_path(dst_path, file_to_transfer)
    # dst_path_full = f'"{dst_full_path}"'
    # print(f'checking dst path: {dst_path_full}')

    # os.stat(dst_path_full)

    # from pathlib import Path
    # my_file = Path(dst_full_path)
    # if not my_file.exists():

    test_cmd = f'test -e "{dst_path_full}"'
    # print('Running {}'.format(test_cmd))

    ret = os.system(test_cmd)

    # print('ret {}'.format(ret))

    if ret:
        # if not os.path.exists(dst_path_full):
        print(f'\n\ntransfer failed:\n{file_to_transfer}\n\n')
        return
    # else:
    #     print(f'transfer succeeded')

    if log_file:
        from datetime import datetime
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

        with open(log_file, 'a') as log_fid:
            log_fid.write(f'# {time_stamp}\n')
            log_fid.write(f'{file_to_transfer}\n')

    if mode == 1 or mode == -2:
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
        'info_root': '',
        'info_dir': '',
        'info_file': '',
        'auth_file': '',
        'port': '',
        'dst_path': '.',
        'scp_path': '.',
        'scp_name': '',
        'src_info': '',
        'log_file': '',
        'ahk_cmd': 'paste_with_cat_1',
    }
    paramparse.process_dict(params)

    win_title = params['win_title']
    use_ahk = params['use_ahk']
    mode = params['mode']
    dst_path = params['dst_path']
    scp_path = params['scp_path']
    scp_name = params['scp_name']
    src_info = params['src_info']
    # port = params['port']

    auth_file = params['auth_file']

    key_root = params['key_root']
    key_dir = params['key_dir']

    info_root = params['info_root']
    info_dir = params['info_dir']
    info_file = params['info_file']
    log_file = params['log_file']

    ahk_cmd = params['ahk_cmd']

    # script_filename = inspect.getframeinfo(inspect.currentframe()).filename
    # script_path = os.path.dirname(os.path.abspath(script_filename))
    #
    # log_dir = os.path.join(script_path, 'log')
    # if not os.path.isdir(log_dir):
    #     os.makedirs(log_dir)
    # print('Saving log to {}'.format(log_dir))

    if not src_info:
        src_info = scp_name

    info_path = linux_path(info_root, info_dir, info_file)

    info_data = open(info_path, 'r').readlines()
    info_data = [k.strip() for k in info_data]

    dst_info = {}

    for datum in info_data:
        info = datum.split(' ')
        name0, name1, dst = info[:3]

        # print(f'info: {info}')

        ecr = key = port = None

        if len(info) > 3:
            port = info[3]

        if len(info) > 4:
            ecr = info[4]

        if len(info) > 5:
            key = info[5]

        dst_info[name0] = [name0, name1, dst, ecr, key, port]

    # print(f'dst_info:\n{dst_info}')

    scp_dst = dst_info[scp_name][2]
    port = dst_info[scp_name][5]

    if mode == -1 or mode == -2:
        """private key based"""
        pwd = auth_file
    else:
        """password based"""
        encryption_params = encryption.Params()
        encryption_params.mode = 1
        encryption_params.root_dir = key_root
        encryption_params.parent_dir = key_dir

        encryption_params.in_file = dst_info[scp_name][-3]
        encryption_params.key_file = dst_info[scp_name][-2]
        encryption_params.process()
        pwd = encryption.run(encryption_params)

    if mode == 0 or mode == -1:
        data_type = 'filename (from {})'.format(src_info)
    elif mode == 1 or mode == -2:
        data_type = 'filename (to {})'.format(src_info)
    elif mode == 2:
        data_type = 'log'

    while True:
        k = input('\nEnter {}\n'.format(data_type))

        if not k:
            continue

        x, y = win32api.GetCursorPos()
        # EnumWindows(EnumWindowsProc(foreach_window), 0)
        if use_ahk:
            if log_file:
                already_transferred = open(log_file, 'r', encoding="utf-8").read().splitlines()
                already_transferred = [_line for _line in already_transferred
                                       if not _line.startswith('#') and _line]
            else:
                already_transferred = []

            if k == '__all__':
                assert log_file, "log_file must be provided to transfer all files"

                list_dir = os.path.dirname(log_file)

                timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
                list_fname = linux_path(f'{src_info}_{timestamp}.txt')

                list_path = os.path.join(list_dir, list_fname)

                ls_cmd = f'ssh {scp_dst} ls {scp_path} > {list_fname}'
                print(f'running: {ls_cmd}')
                os.system(ls_cmd)

                shutil.move(list_fname, list_path)

                all_downloads = open(list_path, 'r', encoding="utf-8").read().splitlines()

                files_to_transfer = list(set(all_downloads) - set(already_transferred))

                # os.system(f'rm {list_fname}')
            elif k == '__list__':
                list_fname = f'{src_info}.txt'
                if not os.path.exists(list_fname):
                    print(f'list file does not exist: {list_fname}')
                files_to_transfer = open(list_fname, 'r', encoding="utf-8").read().splitlines()
            else:
                if k in already_transferred:
                    k2 = input(f'{k} has already been transferred. Transfer again ?\n')
                    if k2.lower() != 'y':
                        continue

                files_to_transfer = [k, ]

            n_files = len(files_to_transfer)
            if n_files == 0:
                print('no files to transfer')

            files_to_transfer.sort()

            files_to_transfer_txt = '\n'.join(files_to_transfer)
            print(f'transferring {n_files} files:\n{files_to_transfer_txt}')

            if mode == 0 or mode == -1:
                clip_txt = f'from {src_info}:\n{files_to_transfer_txt}'
            elif mode == 1 or mode == -2:
                clip_txt = f'to {src_info}:\n{files_to_transfer_txt}'

            try:
                import pyperclip

                pyperclip.copy(clip_txt)
                _ = pyperclip.paste()
            except BaseException as e:
                print('Copying to clipboard failed: {}'.format(e))
            else:
                os.system(ahk_cmd)

            for file_id, file_to_transfer in enumerate(files_to_transfer):
                print(f'{file_id + 1} / {n_files} : {file_to_transfer}')

                run_scp(dst_path, pwd, scp_dst, scp_path, file_to_transfer, mode, port, log_file)

            try:
                import pyperclip

                pyperclip.copy('{}'.format(k))
                _ = pyperclip.paste()
            except:
                pass

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
            run_scp(dst_path, pwd, scp_dst, scp_path, k, mode, port)
            continue

        target_title = target_title[0]
        # print('target_title: {}'.format(target_title))

        try:
            app = application.Application().connect(title=target_title, found_index=0)
        except BaseException as e:
            print('Failed to connect to app for window {}: {}'.format(target_title, e))
            run_scp(dst_path, pwd, scp_dst, scp_path, k, mode, port)
            continue
        try:
            app_win = app.window(title=target_title)
        except BaseException as e:
            print('Failed to access app window for {}: {}'.format(target_title, e))
            run_scp(dst_path, pwd, scp_dst, scp_path, k, mode, port)
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
                app_win.type_keys("{LEFT}{RIGHT}{VK_SPACE}to{VK_SPACE}%s" % src_info)
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

        run_scp(dst_path, pwd, scp_dst, scp_path, k, mode, port)


if __name__ == '__main__':
    main()
