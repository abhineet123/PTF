import re
import paramparse
try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk

from pprint import pformat

def main():
    _params = {
        'field_id': 1,
        'field_sep': '\t',

    }
    paramparse.process_dict(_params)
    field_id = _params['field_id']
    field_sep = _params['field_sep']

    try:
        in_txt = Tk().clipboard_get()
    except BaseException as e:
        print('Tk().clipboard_get() failed: {}'.format(e))
        lines = []
    else:
        lines = in_txt.split('\n')
        lines = [line for line in lines if line.strip()]

    proc_lines = []

    # print(f'lines: {pformat(lines)}')

    for line in lines:
        if field_id >= 0:
            line_fields = line.strip().split(field_sep)
            in_line = line_fields[field_id]
        else:
            in_line = line.strip()

        print(f'in_line: {pformat(in_line)}')

        nums = re.finditer(r'\d+', in_line)
        out_line = in_line
        offset = 0
        for num in nums:
            start_idx, end_idx = num.start(0),  num.end(0)
            token = in_line[start_idx:end_idx]
            out_line = out_line[:start_idx+offset] + '{:04d}'.format(int(token)) + out_line[end_idx+offset:]
            offset = len(out_line) - len(in_line)
            # out_line = out_line.replace(num, '{:04d}'.format(int(num)), 1)

            print(f'token: {pformat(token)}')
            print(f'out_line: {pformat(out_line)}')

        proc_lines.append(out_line)

    print(f'proc_lines: {pformat(proc_lines)}')

    sort_idx = [i[0] for i in sorted(enumerate(proc_lines), key=lambda x: x[1])]

    print(f'sort_idx: {pformat(sort_idx)}')

    out_lines = [lines[idx] for idx in sort_idx]

    out_txt = '\n'.join(out_lines)

    print(out_txt)
    # with open(out_fname, 'w') as out_fid:
    #     out_fid.write(out_txt)
    try:
        import pyperclip

        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


if __name__ == '__main__':
    main()
