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

        self.win_titles = ['Timing', 'Google Chrome']
        self.txt_path = 'C:/Users/Tommy/Documents/Backup/txtpad'
        self.txt_proc_list = 'processed.log'
        self.recursive = 1
        self.ogg_log_path = 'log/ptd_ogg.txt'
        self.category = 2

        self.ffs = Params.FFS()
        self.cmd = Params.CMD()

    class FFS:
        def __init__(self):
            self.enable = 1
            self.exe = ''
            self.root = ''
            self.ext = ''
            self.files = []

    class CMD:
        def __init__(self):
            self.type = ''
            self.paste = ''
            self.link = ''
            self.down = ''
            self.enter = ''


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
    total_t_line = None
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
        elif line.startswith('Total Time: '):
            total_t_line = line.replace('Total Time: ', '').strip()

    if not started:
        raise AssertionError('invalid string to extract timing data from')

    out_txt_lines = [k for k in out_txt.split('\n') if k]

    n_out_txt_lines = len(out_txt_lines)

    print('out_txt_lines: {}'.format(out_txt_lines))
    print('n_out_txt_lines: {}'.format(n_out_txt_lines))
    print('total_t_line: {}'.format(total_t_line))

    if n_out_txt_lines == 1 and total_t_line is not None:
        total_t_line_data = total_t_line.split('.')
        # print('_line_data: {}'.format(_line_data))

        _line = '{}:{}'.format(total_t_line_data[0], int(total_t_line_data[1]) * 10000)
        # print('_line: {}'.format(_line))

        total_t = datetime.strptime(_line, '%H:%M:%S:%f')
        total_t_str = total_t.strftime('%H:%M:%S') + '.{}'.format(total_t_line_data[1])
        curr_t = start_t + timedelta(hours=total_t.hour, minutes=total_t.minute,
                                     seconds=total_t.second, microseconds=total_t.microsecond)
        curr_t_str = curr_t.strftime('%H:%M:%S') + '.{}'.format(int(curr_t.microsecond / 10000))

        out_txt += 'Lap 1\t{}\t{}\t{}\n'.format(total_t_str, total_t_str, curr_t_str)

    return out_txt


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

    # win32clipboard.CloseClipboard()


def process_ogg(ogg_paths, lines, category, is_path, cmd, down_to_end, ogg_log_path):
    """

    :param ogg_paths:
    :param lines:
    :param category:
    :param is_path:
    :type cmd: Params.CMD
    :return:
    """
    base_names = [os.path.basename(line) for line in ogg_paths]
    out_lines = [base_name.split('-')[1].split('.')[0].replace('_', ':') for base_name in base_names]
    if category > 0:
        out_lines = ['{} :: {}'.format(line, category) for line in out_lines]

    sort_idx = sorted(range(len(out_lines)), key=out_lines.__getitem__)

    out_lines = [out_lines[k] for k in sort_idx]
    lines = [lines[k] for k in sort_idx]

    if out_lines:
        out_txt = '\n'.join(out_lines)
        if cmd.type:
            # type_file = cmd.type.replace('exe', 'txt')
            type_file = cmd.type + '.txt'
            with open(type_file, 'w') as fid:
                fid.write(out_txt)
            os.system(cmd.type)
        else:
            copy_to_clipboard(out_txt, print_txt=1)
            time.sleep(1.0)
            os.system(cmd.paste)

    # in_txt = '\n'.join(lines)
    # print(in_txt)
    # if pause_for_input:
    #     input('press any key')

    # print('is_path: {}'.format(is_path))
    # print('paste: {}'.format(cmd.paste))
    # print('link: {}'.format(cmd.link))
    # print('down: {}'.format(cmd.down))
    # print('enter: {}'.format(cmd.enter))

    ogg_log_dir = os.path.dirname(ogg_log_path)
    os.makedirs(ogg_log_dir, exist_ok=True)

    if is_path and cmd.link and lines:
        log_fid = open(ogg_log_path, 'a')
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        log_fid.write(f'\n# {timestamp}\n')
        for _path in lines[::-1]:
            print(_path)
            log_fid.write(_path + '\n')
            if cmd.type:
                # type_file = cmd.link.replace('exe', 'txt')
                type_file = cmd.link + '.txt'
                with open(type_file, 'w') as fid:
                    fid.write(_path)
                os.system(cmd.link)
            else:
                copy_to_clipboard(_path, print_txt=1)
                time.sleep(0.5)
                # input('press any key')
                os.system(cmd.link)
                time.sleep(0.5)
        log_fid.close()

        if down_to_end and is_path and cmd.down and cmd.enter:
            for _ in range(len(lines)):
                os.system(cmd.down)
                os.system(cmd.down)
            os.system(cmd.enter)


def filter_ogg(ogg_log_path, ogg_paths, ogg_lines):
    if os.path.exists(ogg_log_path):
        processed_ogg = open(ogg_log_path, 'r').readlines()
        processed_ogg = [line.strip() for line in processed_ogg if line.strip() and not line.startswith('#')]
        valid_ogg_ids = [i for i, line in enumerate(ogg_lines) if line not in processed_ogg]
        ogg_paths = [ogg_paths[i] for i in valid_ogg_ids]
        ogg_lines = [ogg_lines[i] for i in valid_ogg_ids]

    return ogg_paths, ogg_lines


def main():
    params = Params()
    paramparse.process(params)

    in_txt = copy_from_clipboard()
    if in_txt is None:
        lines = []
    else:
        lines = in_txt.split('\n')
        lines = [line.strip() for line in lines if line.strip()]

    if lines:
        is_path = all(line.startswith('"') and line.endswith('"') for line in lines)
        if is_path:
            stripped_lines = [line.strip('"') for line in lines]
        else:
            stripped_lines = lines[:]

        try:
            is_ogg = all(line.endswith('.ogg') for line in stripped_lines)
            is_folder = all(os.path.isdir(line) for line in stripped_lines)
        except BaseException as e:
            print('exception during ogg / folder check : {}'.format(e))
            is_ogg = 0
            is_folder = 0

        print('is_ogg: {}'.format(is_ogg))
        print('stripped_lines: {}'.format(stripped_lines))
        print('lines: {}'.format(lines))
        #
        # input('press any key')

        if is_ogg:
            stripped_lines, lines = filter_ogg(params.ogg_log_path, stripped_lines, lines)
            process_ogg(stripped_lines, lines, params.category, is_path, params.cmd,
                        down_to_end=0, ogg_log_path=params.ogg_log_path)
            return
        elif is_folder:
            n_stripped_lines = len(stripped_lines)
            stripped_lines.sort()
            for folder_id, folder in enumerate(stripped_lines):
                ogg_paths = ['{}'.format(os.path.join(folder, k)) for k in os.listdir(folder) if k.endswith('.ogg')]
                ogg_lines = ['"{}"'.format(k) for k in ogg_paths]

                ogg_paths, ogg_lines = filter_ogg(params.ogg_log_path, ogg_paths, ogg_lines)
                process_ogg(ogg_paths, ogg_lines, params.category, is_path, params.cmd,
                            down_to_end=folder_id < n_stripped_lines - 1, ogg_log_path=params.ogg_log_path)
            return
        else:
            try:
                out_txt = process(in_txt)
            except:
                pass
            else:
                copy_to_clipboard(out_txt, print_txt=1)
                time.sleep(0.5)
                return

    if params.ffs.enable:
        for _ffs_file in params.ffs.files:
            ffs_path = os.path.join(params.ffs.root, _ffs_file + '.' + params.ffs.ext)
            ffs_cmd = '{} "{}"'.format(params.ffs.exe, ffs_path)
            print(ffs_cmd)
            os.system(ffs_cmd)

    if params.txt_path:

        assert os.path.isdir(params.txt_path), "invalid text path: {}".format(params.txt_path)

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

        if n_files > 0:
            _ = input('\nfound {} new files:\n{}\nPress any key to continue\n'.format(n_files, files))
        else:
            _ = input('\nfound no new files. Press any key to exit\n')

        for file_id, file in enumerate(files[::-1]):
            if file_id > 0:
                _ = input('\nDone {} / {}. Press any key to continue\n'.format(file_id, n_files))

            print('reading file {} / {}: {}'.format(file_id + 1, n_files, file))

            # file = dst_file

            in_txt = open(file, 'r').read()
            out_txt = process(in_txt, verbose=0)
            print(out_txt)

            copy_to_clipboard(out_txt)
            time.sleep(0.5)

            out_txt_lines = [k for k in out_txt.split('\n') if k]

            n_out_txt_lines = len(out_txt_lines)

            # dst_file = file.replace('.txt', '.log')
            # shutil.move(file, dst_file)

            print('out_txt_lines: {}'.format(out_txt_lines))
            print('n_out_txt_lines: {}'.format(n_out_txt_lines))

            if n_out_txt_lines == 1:
                os.system("vscode {}".format(file))

            with open(txt_proc_list_path, 'r+') as f:
                content = f.read()
                f.seek(0, 0)

                timestamp_str = datetime.now().strftime("%y%m%d %H:%M:%S.%f")[:-4]

                txt = '{}\t{}\n'.format(timestamp_str, file)
                f.write(txt + content)

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

    in_txt = copy_from_clipboard()
    out_txt = process(in_txt)

    # with open(out_fname, 'w') as out_fid:
    #     out_fid.write(out_txt)

    copy_to_clipboard(out_txt, print_txt=1)


if __name__ == '__main__':
    main()
