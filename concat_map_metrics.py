from datetime import datetime
import os
import json
import paramparse

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk

class Params:

    def __init__(self):
        self.list_path = 'list'
        self.out_path = 'list_out'
        self.json_name = 'eval_result_dict.json'
        self.class_name = 'ipsc'
        self.metrics = ['FNR_DET', 'FN_DET', 'FPR_DET', 'FP_DET']
        self.sep = '\t'


def main():
    params = Params()
    paramparse.process(params)

    in_dirs = []
    try:
        in_txt = Tk().clipboard_get()  # type: str
    except BaseException as e:
        print('Tk().clipboard_get() failed: {}'.format(e))
    else:
        in_dirs = [k.strip() for k in in_txt.split('\n') if k.strip()]

    if not in_dirs or not all(os.path.isdir(in_dir) for in_dir in in_dirs):
        in_dirs = open(params.list_path, 'r').read().splitlines()

    out_lines = {}
    for metric in params.metrics:
        header = f'model\t{metric}'
        out_lines[metric] = [header, ]

    for in_dir in in_dirs:
        in_dir_name = os.path.basename(in_dir)
        json_path = os.path.join(in_dir, params.json_name)
        with open(json_path, 'r') as fid:
            json_data = json.load(fid)

        for metric in params.metrics:
            if params.class_name:
                metric_val = json_data[params.class_name][params.metric]
            else:
                metric_val = json_data[params.metric]

            out_line = f'{in_dir_name}\t{metric_val:f}'
            out_lines[metric].append(out_line)

    out_txt = '\n'.join(out_lines)

    open(params.out_path, 'w').write(out_txt)

    # print('out_txt:\n{}'.format(out_txt))

    try:
        import pyperclip

        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


if __name__ == '__main__':
    main()
