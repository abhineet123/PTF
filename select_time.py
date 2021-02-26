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
        'horz': 1,
        'categories': 1,
        'category_sep': ' :: ',
        'pairwise': 1,
        'first_and_last': 0,
        'add_date': 1,

    }
    paramparse.process_dict(_params)
    horz = _params['horz']
    categories = _params['categories']
    category_sep = _params['category_sep']
    first_and_last = _params['first_and_last']
    pairwise = _params['pairwise']
    add_date = _params['add_date']

    try:
        in_txt = Tk().clipboard_get()  # type: str
    except BaseException as e:
        print('Tk().clipboard_get() failed: {}'.format(e))
        return

    lines = in_txt.split('\n')
    out_lines = []
    out_categories = []

    date_str = datetime.now().strftime("%y-%m-%d")

    for line in lines:
        category = None
        if categories:
            if category_sep in line:
                temp = line.split(category_sep)[:2]

                print('line: {}'.format(line))
                print('temp: {}'.format(temp))

                if len(temp) == 2:
                    line, category = temp
                    line = line.strip()
                    category = category.strip()
        try:
            datetime.strptime(line, '%I:%M:%S %p')
        except ValueError:
            pass
        else:
            out_lines.append(line)
            if categories:
                if category is None:
                    category = -1
                out_categories.append(category)

    if first_and_last and len(out_lines) > 2:
        out_lines = [out_lines[0], out_lines[-1]]

    n_out_lines = len(out_lines)
    if pairwise:
        out_txt0 = ''
        out_txt = ''
        out_txt2 = ''
        out_txt3 = ''
        for _line_id in range(n_out_lines - 1):
            if horz:
                _out_txt = '{}\t{}'.format(out_lines[_line_id], out_lines[_line_id + 1])
                if add_date:
                    _out_txt = '{}\t{}'.format(date_str, _out_txt)
                if categories:
                    _out_txt += '\t{}'.format(out_categories[_line_id + 1])
                out_txt += _out_txt + '\n'
            else:
                out_txt += '{}\t'.format(out_lines[_line_id])
                out_txt2 += '{}\t'.format(out_lines[_line_id + 1])
                if add_date:
                    out_txt0 = '{}\t'.format(date_str)
                if categories:
                    out_txt3 += '{}\t'.format(out_categories[_line_id + 1])
        if not horz:
            out_txt += '\n' + out_txt2
            if add_date:
                out_txt = '{}\n{}'.format(out_txt0, out_txt)
            if categories:
                out_txt += '\n' + out_txt3
    else:
        if horz:
            field_sep = '\t'
        else:
            field_sep = '\n'

        out_txt = field_sep.join(out_lines)

    out_txt = out_txt.rstrip()

    print('out_txt:\n{}'.format(out_txt))

    try:
        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


if __name__ == '__main__':
    main()
