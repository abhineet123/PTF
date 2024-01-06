from datetime import datetime
import paramparse

import win32clipboard
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
        'category': 2,
        'category_sep': ' :: ',
        'date_sep': ' – ',
        'pairwise': 1,
        'first_and_last': 0,
        'add_date': 1,
        'add_diff': 1,

    }
    paramparse.process_dict(_params)
    category = _params['category']
    category_sep = _params['category_sep']

    try:
        win32clipboard.OpenClipboard()
        in_txt = win32clipboard.GetClipboardData()  # type: str
    except BaseException as e:
        print('Tk().clipboard_get() failed: {}'.format(e))
        win32clipboard.CloseClipboard()
        return
    else:
        win32clipboard.CloseClipboard()

    lines = in_txt.split('\n')
    out_lines = []

    for line in lines:
        in_category = None
        if category_sep in line:
            temp = line.split(category_sep)

            # print('line: {}'.format(line))
            # print('temp: {}'.format(temp))

            if len(temp) == 2:
                line, in_category = temp
                line = line.strip()
                in_category = int(in_category.strip())

        line, time_found, time_obj = is_time(line)

        if time_found:
            if in_category is not None and in_category != category:
                print('replacing category {}  with {} in {}'.format(in_category, category, line))

            line = '{} :: {}'.format(line, category)

        out_lines.append(line)

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
