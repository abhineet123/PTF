from datetime import datetime
import paramparse

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk

import pyperclip


def main():
    _params = {
        'field_id': 2,
        'field_sep': ' ',

    }
    paramparse.process_dict(_params)

    try:
        in_txt = Tk().clipboard_get()  # type: str
    except BaseException as e:
        print('Tk().clipboard_get() failed: {}'.format(e))
        return

    lines = in_txt.split('\n')
    out_lines = []
    for line in lines:
        try:
            datetime.strptime(line, '%I:%M:%S %p')
        except ValueError:
            pass
        else:
            out_lines.append(line)

    out_txt = '\n'.join(out_lines)
    print('out_txt: {}'.format(out_txt))

    try:
        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


if __name__ == '__main__':
    main()
