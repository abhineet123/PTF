from datetime import datetime, timedelta
import paramparse
from urllib.parse import unquote

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk

class Params:

    def __init__(self):
        self.add_comment = 1
        self.add_date = 1
        self.add_diff = 1
        self.categories = 1
        self.categories_out = 1
        self.category_sep = ' :: '
        self.date_sep = ' – '
        self.first_and_last = 0
        self.horz = 1
        self.included_cats = []
        self.min_start_time = '03:00:00'
        self.pairwise = 1


def main():
    _params = Params()
    paramparse.process(_params)

    try:
        in_txt = Tk().clipboard_get()  # type: str
    except BaseException as e:
        print('Tk().clipboard_get() failed: {}'.format(e))
        return

    lines = [k.strip() for k in in_txt.split('\n') if k.strip()]

    names = []
    for line in lines:
        line_parts = line.split("&dn=")
        line_parts2 = line_parts[1].split("&tr=")

        name_str = line_parts2[0].replace("%20", " ")

        name_str2 = unquote(line_parts2[0])
        print(name_str)

        names.append(name_str2)

    out_txt = '\n'.join(names)

    print('out_txt:\n{}'.format(out_txt))

    try:
        import pyperclip

        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


if __name__ == '__main__':
    main()
