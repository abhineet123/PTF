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
        'horz': 0,
        'first_and_last': 0,

    }
    paramparse.process_dict(_params)
    horz = _params['horz']
    first_and_last = _params['first_and_last']

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

    if horz:
        field_sep = '\t'
    else:
        field_sep = '\n'

    if first_and_last and len(out_lines) > 2:
        out_lines = [out_lines[0], out_lines[-1]]

    out_txt = field_sep.join(out_lines)
    print('out_txt: {}'.format(out_txt))

    try:
        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


if __name__ == '__main__':
    main()
