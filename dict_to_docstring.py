import paramparse
# import re
from difflib import SequenceMatcher
from pprint import pformat

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk

from ast import literal_eval
import pyperclip


def insert_newlines_by_matching(in_line_id, in_lines, out_line):
    len_out_line = len(out_line)

    while in_line_id < len(in_lines):
        in_line_id += 1

        in_line = in_lines[in_line_id]
        len_in_line = len(in_line)

        print(f'in_line: {in_line}')
        print(f'len_in_line: {len_in_line}')
        print(f'out_line: {out_line}')

        match = SequenceMatcher(None, in_line, out_line).find_longest_match(
            0, len_in_line, 0, len_out_line)

        match_str = in_line[match.a: match.a + match.size]
        match_ratio = float(match.size) / float(len_out_line)

        print(f'match_str: {match_str}')
        print(f'match_ratio: {match_ratio}')
        print()

        if match_ratio > 0.5:
            break


def insert_newlines(string, every=70):
    lines = []
    start = 0
    while start < len(string):
        end = start + every - 1
        while end < len(string) and string[end] != ' ':
            end += 1
        lines.append(string[start:end + 1])
        start = end + 1
    return '\n        '.join(lines)


def main():
    _params = {
        'break_lines': 1,
        'field_sep': ' ',

    }
    paramparse.process_dict(_params)
    break_lines = _params['break_lines']
    field_sep = _params['field_sep']

    try:
        in_txt = Tk().clipboard_get()
    except BaseException as e:
        print('Tk().clipboard_get() failed: {}'.format(e))
        return

    in_dict = literal_eval(in_txt)
    out_txt = ''
    # in_lines = in_txt.splitlines()
    # in_line_id = -1

    out_txt2 = in_txt.strip().lstrip('{').rstrip('}').replace('                ', '        ')

    for _var_name in in_dict:
        # out_line = '{}'.format(in_dict[_var_name])

        new_line = ':ivar {}: {}'.format(_var_name, in_dict[_var_name])
        new_line_broken = new_line

        out_txt2 = out_txt2.replace("'{}': ".format(_var_name), ':ivar {}: '.format(_var_name))

        if break_lines:
            # new_line_broken = re.sub("(.{70})", "\\1\n", new_line, 0, re.DOTALL)
            # new_line_broken = insert_newlines_by_matching(in_line_id, in_lines, out_line)
            new_line_broken = insert_newlines(new_line, 70)

        out_txt += '        ' + new_line_broken + '\n\n'

    print('out_txt:\n{}'.format(out_txt))
    print('out_txt2:\n{}'.format(out_txt2))

    # out_txt_final = out_txt + '\n\n' + out_txt2

    try:
        pyperclip.copy(out_txt2)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


if __name__ == '__main__':
    main()
