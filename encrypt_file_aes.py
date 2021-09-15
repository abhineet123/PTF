from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes

import os
import time

import paramparse


class Params:
    def __init__(self):
        self.root_dir = ""
        self.root_dir_key = ""
        self.root_dir_out = ""

        self.parent_dir = ""
        self.parent_dir_key = ""
        self.parent_dir_out = ""

        self.in_file = "in.in"
        self.key_file = "key.key"
        self.out_file = "out.out"
        self.mode = 0
        self.from_clipboard = 0
        self.clipboard = 1
        self.auto_switch = 0

    def process(self):
        if not self.root_dir_key:
            self.root_dir_key = self.root_dir

        if not self.parent_dir_key:
            self.parent_dir_key = self.parent_dir

        if not self.root_dir_out:
            self.root_dir_out = self.root_dir

        if not self.parent_dir_out:
            self.parent_dir_out = self.parent_dir

        self.in_file = os.path.join(self.root_dir, self.parent_dir, self.in_file)
        self.key_file = os.path.join(self.root_dir_key, self.parent_dir_key, self.key_file)
        self.out_file = os.path.join(self.root_dir_out, self.parent_dir_out, self.out_file)


def write_key(key_file):
    """
    Generates a key and save it into a file
    """
    key = get_random_bytes(32)
    with open(key_file, "wb") as fid:
        fid.write(key)


def load_key(key_file):
    """
    Loads the key from the current directory named `key.key`
    """
    return open(key_file, "rb").read()


def encrypt(filename, key, out_file, clipboard):
    """
    Given a filename (str) and key (bytes), it encrypts the file and write it
    """
    cipher = AES.new(key, AES.MODE_EAX)

    if clipboard:
        try:
            from Tkinter import Tk
        except ImportError:
            from tkinter import Tk
        try:
            in_txt = Tk().clipboard_get()
        except BaseException as e:
            raise AssertionError('Tk().clipboard_get() failed: {}'.format(e))
        file_data = in_txt.encode('ascii')
    else:
        file_data = open(filename, "rb").read()

    # encrypt data
    ciphertext, tag = cipher.encrypt_and_digest(file_data)

    # write the encrypted file
    with open(out_file, "wb") as file:
        [file.write(x) for x in (cipher.nonce, tag, ciphertext)]

    return ciphertext, tag


def decrypt(filename, key):
    """
    Given a filename (str) and key (bytes), it decrypts the file and write it
    """
    with open(filename, "rb") as file:
        # read the encrypted data
        nonce, tag, ciphertext = [file.read(x) for x in (16, 16, -1)]

    cipher = AES.new(key, AES.MODE_EAX, nonce)
    # decrypt data
    decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)

    return decrypted_data


def find_last_active_window():
    import win32gui

    win_titles = []
    win_handles = []

    def foreach_window(hwnd, lParam):
        import ctypes

        GetWindowText = ctypes.windll.user32.GetWindowTextW
        GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
        IsWindowVisible = ctypes.windll.user32.IsWindowVisible

        if IsWindowVisible(hwnd):
            length = GetWindowTextLength(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            win_titles.append(buff.value)
            win_handles.append(hwnd)
        return True

    win32gui.EnumWindows(foreach_window, None)

    vwm_title_start = 'VWM_'

    target_id = [i for i, k in enumerate(win_titles) if k.startswith(vwm_title_start)]
    if not target_id:
        print('window with title starting with "{}"" not found'.format(vwm_title_start))
        return False

    if len(target_id) > 1:
        target_win_titles = [win_titles[i] for i in target_id]
        print(
            'multiple windows with titles starting with "{}"" found: {}'.format(vwm_title_start, target_win_titles))
        return False

    vwm_title = win_titles[target_id[0]]
    vwm_handle = win_handles[target_id[0]]

    print('\nfound vwm window with title {} and handle {}\n'.format(vwm_title, vwm_handle))

    if vwm_handle is not None:
        import win32api
        import win32con
        win32api.PostMessage(vwm_handle, win32con.WM_CHAR, 0x45, 0)

        time.sleep(1)

        last_active_handle = copy_from_clipboard()

        try:
            last_active_handle_int = int(last_active_handle)
        except TypeError:
            print('invalid last_active_handle: {}'.format(last_active_handle))
            return False

        last_active_name = win32gui.GetWindowText(last_active_handle_int)

        print('last_active_handle_int: {}'.format(last_active_handle_int))
        print('last_active_name: {}'.format(last_active_name))
        win32gui.SetForegroundWindow(last_active_handle_int)
        return True


def type_string(out_txt, auto_switch):
    if auto_switch:
        find_last_active_window()

    else:
        # print('waiting 1 second to change active app')
        time.sleep(1)

    import pyautogui
    pyautogui.write(out_txt)
    pyautogui.press('enter')

    # for c in out_txt:
    #     keyboard.send(c)


def copy_from_clipboard():
    try:
        import win32clipboard

        win32clipboard.OpenClipboard()
        in_txt = win32clipboard.GetClipboardData()
    except BaseException as e:
        print('Copying from clipboard failed: {}'.format(e))
        win32clipboard.CloseClipboard()
        return None
    win32clipboard.CloseClipboard()
    return in_txt


def copy_to_clipboard(out_txt):
    try:
        import pyperclip

        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


def run(params):
    """

    :param Params params:
    :return:
    """

    if params.mode == 0:
        write_key(params.key_file)
        # load the key
        key = load_key(params.key_file)

        # encrypt it
        encrypt(params.in_file, key, params.out_file, params.clipboard)
    else:

        if params.from_clipboard:
            out_txt = copy_from_clipboard()
            print('out_txt: {}'.format(out_txt))
        else:
            key = load_key(params.key_file)
            decrypted_data = decrypt(params.in_file, key)
            out_txt = decrypted_data.decode('ascii')

        if params.clipboard:
            if params.clipboard == 1:
                copy_to_clipboard(out_txt)
            elif params.clipboard == 2:
                type_string(out_txt, params.auto_switch)
        else:
            # write the original file
            with open(params.out_file, "wb") as file:
                file.write(decrypted_data)

        return out_txt


if __name__ == '__main__':
    _params = Params()
    paramparse.process(_params)
    _params.process()

    run(_params)
