import re
import numpy as np

import paramparse

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk

from pprint import pformat
import pyperclip
import pyautogui


def is_number(num_str):
    try:
        k = float(num_str)
    except:
        return False

    return True


def main():
    _params = {
        'field_id': 1,
        'field_sep': '\t',
        'token_sep': '/',
        'inverted': 1,
        'remove_duplicates': 1,
        'max_cols': 7,

        # 'mismatch_replace': [],
        'mismatch_replace': ['abs', 'rand'],

    }
    paramparse.process_dict(_params)
    field_id = _params['field_id']
    field_sep = _params['field_sep']
    token_sep = _params['token_sep']
    inverted = _params['inverted']
    remove_duplicates = _params['remove_duplicates']
    max_cols = _params['max_cols']

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
        lines_list = [line.strip().split(field_sep) for line in lines if line.strip()]

    n_lines = len(lines_list)

    assert n_lines > 1, "too few lines to analyse"

    numerical_column_ids = [i for i, val in enumerate(lines_list[0]) if is_number(val)]

    n_cols = len(numerical_column_ids)

    if max_cols < n_cols:
        numerical_column_ids = numerical_column_ids[:max_cols]
        n_cols = max_cols

    numerical_data = [
        [float(line[i]) for i in numerical_column_ids]
        for line in lines_list
    ]

    numerical_data = np.array(numerical_data)

    max_ids = np.argmax(numerical_data, axis=0)
    min_ids = np.argmin(numerical_data, axis=0)

    max_vals = np.amax(numerical_data, axis=0)
    min_vals = np.amin(numerical_data, axis=0)
    mean_vals = np.mean(numerical_data, axis=0)
    median_vals = np.median(numerical_data, axis=0)

    max_lines = [lines[i] for i in max_ids]
    min_lines = [lines[i] for i in min_ids]

    max_vals_str = '\t'.join(str(k) for k in max_vals)
    min_vals_str = '\t'.join(str(k) for k in min_vals)
    mean_vals_str = '\t'.join(str(k) for k in mean_vals)
    median_vals_str = '\t'.join(str(k) for k in median_vals)

    out_txt = '\n'.join([max_vals_str, min_vals_str, mean_vals_str, median_vals_str])

    out_txt += '\n\n' + '\n'.join(max_lines) + '\n\n' + '\n'.join(min_lines)

    try:
        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


if __name__ == '__main__':
    main()
