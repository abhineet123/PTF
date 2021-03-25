from datetime import datetime
import paramparse

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk

import pyperclip

def is_time(line):
    time_found = 0
    time_obj = None
    try:
        time_obj = datetime.strptime(line, '%I:%M:%S %p')
    except ValueError:
        try:
            time_obj = datetime.strptime(line, '%I:%M %p')
        except ValueError:
            try:
                time_obj = datetime.strptime(line, '%H:%M:%S')
            except ValueError:
                pass
            # else:
        # else:
        #     temp2 = line.split(' ')
        #     _time, _pm = temp2
        #     line = '{}:00 {}'.format(_time, _pm)
        #     time_found = 1
    # else:
    #     time_found = 1

    if time_obj is not None:
        line = time_obj.strftime('%I:%M:%S %p')
        time_found = 1

    return line, time_found, time_obj


def main():
    _params = {
        'horz': 1,
        'categories': 1,
        'category_sep': ' :: ',
        'date_sep': ' – ',
        'pairwise': 1,
        'first_and_last': 0,
        'add_date': 1,

    }
    paramparse.process_dict(_params)
    horz = _params['horz']
    categories = _params['categories']
    category_sep = _params['category_sep']
    date_sep = _params['date_sep']
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
                temp = line.split(category_sep)

                # print('line: {}'.format(line))
                # print('temp: {}'.format(temp))

                if len(temp) == 2:
                    line, category = temp
                    line = line.strip()
                    category = category.strip()

        if date_sep in line:
            temp = line.split(date_sep)

            _, time_found, _ = is_time(temp[0])
            if time_found:
                print('date_sep line: {}'.format(line))
                print('date_sep temp: {}'.format(temp))

                if len(temp) == 3:
                    line, _, date_str = temp
                    line = line.strip()
                elif len(temp) == 2:
                    line, possible_date = temp
                    line = line.strip()
                    _, time_found, _ = is_time(possible_date)
                    if not time_found:
                        date_str = possible_date

        line, time_found, _ = is_time(line)

        if time_found:
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
