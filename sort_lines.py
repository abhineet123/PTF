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
        'field_id': 1,
        'field_sep': '\t',
        'token_sep': '/',
        'inverted': 1,

    }
    paramparse.process_dict(_params)
    field_id = _params['field_id']
    field_sep = _params['field_sep']
    token_sep = _params['token_sep']
    inverted = _params['inverted']

    # pyautogui.hotkey('ctrl', 'c')

    # while True:
    #     _ = input('Press enter to continue\n')

    try:
        in_txt = Tk().clipboard_get()
    except BaseException as e:
        print('Tk().clipboard_get() failed: {}'.format(e))
        return
    else:
        lines = in_txt.split('\n')
        lines = [line for line in lines if line.strip()]

    try:
        proc_lines = []

        # print(f'lines: {pformat(lines)}')

        for line in lines:
            if field_id >= 0:
                line_fields = line.strip().split(field_sep)
                in_line = line_fields[field_id]
            else:
                in_line = line.strip()

            # print(f'in_line: {pformat(in_line)}')

            nums = re.finditer(r'\d+', in_line)
            out_line = in_line
            offset = 0
            for num in nums:
                start_idx, end_idx = num.start(0), num.end(0)
                token = in_line[start_idx:end_idx]
                out_line = out_line[:start_idx + offset] + '{:04d}'.format(int(token)) + out_line[end_idx + offset:]
                offset = len(out_line) - len(in_line)
                # out_line = out_line.replace(num, '{:04d}'.format(int(num)), 1)

                # print(f'token: {pformat(token)}')
                # print(f'out_line: {pformat(out_line)}')

            proc_lines.append(out_line)

        # print(f'proc_lines: {pformat(proc_lines)}')

        # proc_lines_sorted = sorted(proc_lines)
        # proc_lines_txt = '\n'.join(proc_lines)
        # proc_lines_sorted_txt = '\n'.join(proc_lines_sorted)

        if inverted:
            proc_lines_tokens = [k.split(token_sep)[::-1] for k in proc_lines]
        else:
            proc_lines_tokens = [k.split(token_sep) for k in proc_lines]

        sort_idx = [i[0] for i in sorted(enumerate(proc_lines_tokens), key=lambda x: x[1])]

        # print(f'sort_idx: {pformat(sort_idx)}')

        out_lines = [lines[idx] for idx in sort_idx]

        out_txt = '\n'.join(out_lines) + '\n'

        # print(out_txt)
    except BaseException as e:
        print('Processing clipboard contents failed: {}'.format(e))
        return

    try:
        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))

if __name__ == '__main__':
    main()
