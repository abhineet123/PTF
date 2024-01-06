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
        'remove_duplicates': 1,

        # 'mismatch_replace': [],
        'mismatch_replace': ['abs', 'rand'],

    }
    paramparse.process_dict(_params)
    field_id = _params['field_id']
    field_sep = _params['field_sep']
    token_sep = _params['token_sep']
    inverted = _params['inverted']
    remove_duplicates = _params['remove_duplicates']
    mismatch_replace = _params['mismatch_replace']

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

    """Inverting the lines to retain the last occurrence of any duplicate Token in the unique list
     as being the last one and so presumably the correct one
     """
    lines = lines[::-1]

    # try:
    unique_lines = []
    proc_lines = []

    # print(f'lines: {pformat(lines)}')

    in_lines = []
    mismatched_dup_lines = []
    dup_in_lines = []
    dup_in_lines_dict = {}
    for line_id, line in enumerate(lines):
        if field_id >= 0:
            line_fields = line.strip().split(field_sep)
            in_line = line_fields[field_id]
            line_data = line_fields[field_id + 1:]
        else:
            in_line = line.strip()
            line_data = ''

        # print(f'in_line: {pformat(in_line)}')

        if remove_duplicates:
            try:
                dup_line_id, dup_line_data, dup_line = dup_in_lines_dict[in_line]
            except KeyError:
                dup_in_lines_dict[in_line] = (line_id, line_data, line)
            else:
                if not line_data or line_data == dup_line_data:
                    dup_in_lines.append(line)
                    continue
                else:
                    txt = "mismatch in lines {} and {}".format(dup_line_id, line_id)
                    print(txt)
                    # line = txt.replace(' ', '_') + '\t' + line

                    # if mismatch_replace:
                    #     src_txt, dst_txt = mismatch_replace
                    #     if src_txt in line:
                    #         pass

                    try:
                        dup_line_idx = unique_lines.index(dup_line)
                    except ValueError:
                        pass
                    else:
                        del unique_lines[dup_line_idx]
                        del proc_lines[dup_line_idx]

                    mismatched_dup_lines.append([line, dup_line])
                    continue

        unique_lines.append(line)
        in_lines.append(in_line)

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

    """Inverting back"""
    proc_lines = proc_lines[::-1]
    unique_lines = unique_lines[::-1]

    if inverted:
        proc_lines_tokens = [k.split(token_sep)[::-1] for k in proc_lines]
    else:
        proc_lines_tokens = [k.split(token_sep) for k in proc_lines]

    sort_idx = [i[0] for i in sorted(enumerate(proc_lines_tokens), key=lambda x: x[1])]

    # print(f'sort_idx: {pformat(sort_idx)}')

    out_lines = [unique_lines[idx] for idx in sort_idx]

    out_txt = '\n'.join(out_lines) + '\n'

    if mismatched_dup_lines:
        mismatched_dup_lines = mismatched_dup_lines[::-1]

        mismatched_sep_txt = '\n'.join(k[0] for k in mismatched_dup_lines) + '\n\n' + '\n'.join(
            k[1] for k in mismatched_dup_lines)
        out_txt = out_txt + '\n\n' + 'mismatched_sep' + '\n\n' + mismatched_sep_txt

        mismatched_txt = '\n'.join('{}\n{}\n'.format(k[0], k[1]) for k in mismatched_dup_lines)
        out_txt = out_txt + '\n\n' + 'mismatched' + '\n\n' + mismatched_txt + '\n'

    if remove_duplicates and dup_in_lines:
        dup_in_lines_str = '\n'.join(dup_in_lines) + '\n'
        with open('dup_in_lines.txt', 'w') as fid:
            fid.write(dup_in_lines_str)

    # print(out_txt)
    # except BaseException as e:
    #     print('Processing clipboard contents failed: {}'.format(e))
    #     return

    try:
        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


if __name__ == '__main__':
    main()
