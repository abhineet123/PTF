import re
import paramparse

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk

from pprint import pformat
import pyperclip
import pyautogui


def main():
    _params = {
        'field_id': 2,
        'field_sep': ' ',

    }
    paramparse.process_dict(_params)
    field_id = _params['field_id']
    field_sep = _params['field_sep']

    try:
        in_txt = Tk().clipboard_get()
    except BaseException as e:
        print('Tk().clipboard_get() failed: {}'.format(e))
        return

    tokens = in_txt.strip().split(field_sep)

    print('tokens: {}'.format(tokens))
    out_tokens = tokens[field_id:]
    print('out_tokens: {}'.format(out_tokens))

    out_txt = field_sep.join(out_tokens)
    print('out_txt: {}'.format(out_txt))

    try:
        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))

if __name__ == '__main__':
    main()