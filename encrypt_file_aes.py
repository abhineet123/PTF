from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes

import os

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


def type_string(out_txt):
    # import keyboard
    import pyautogui

    import time
    # print('waiting 1 second to change active app')
    time.sleep(1)
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
                type_string(out_txt)
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
