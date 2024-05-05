import os
import paramparse


class Params:
    def __init__(self):
        self.in_fname = ''
        self.out_fname = ''
        self.check_exist = 0
        self.cb = 0


def main():
    params = Params()
    paramparse.process(params, verbose=1)
    out_fname = params.out_fname

    if params.cb:
        try:
            from Tkinter import Tk
        except ImportError:
            from tkinter import Tk
            # import tkinter as Tk

        try:
            in_txt = Tk().clipboard_get()  # type: str
        except BaseException as e:
            print('Tk().clipboard_get() failed: {}'.format(e))
            return

        lines = [k.strip() for k in in_txt.split('\n') if k.strip()]
    else:
        assert params.in_fname, "in_fname must be provided"
        assert os.path.isfile(params.in_fname), f'Input file {params.in_fname:s} does not exist'

        if not out_fname:
            out_fname = f'{params.in_fname}.unique'

        print(f'Removing duplicate lines in {params.in_fname:s} to {out_fname:s}')

        lines = open(params.in_fname, "r").readlines()

    lines = [line.strip() for line in lines if line.strip()]

    # lines = list(set(lines))

    seen = set()
    seen_add = seen.add
    lines = [x for x in lines if not (x in seen or seen_add(x))]

    print(f'Found {len(lines)} unique lines')

    if params.check_exist:
        lines = [line for line in lines if os.path.exists(line)]
        print(f'Found {len(lines)} unique and existent lines')

    out_txt = '\n'.join(lines)

    if params.cb:
        try:
            import pyperclip
            pyperclip.copy(out_txt)
            pyperclip.paste()
        except BaseException as e:
            print('Copying to clipboard failed: {}'.format(e))
    else:
        open(out_fname, "w").write(out_txt)


if __name__ == '__main__':
    main()
