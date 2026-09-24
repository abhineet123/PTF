import os
import sys
import time

import paramparse

from Misc import to_str, copy_from_clipboard


class Params:
    def __init__(self):
        self.cfg = ()
        self.cmd = 'yt-dlp'
        self.list = ''
        self.cb = 1



def main():
    params: Params = paramparse.process(Params)
    assert params.cmd, "params must be provided"

    args_history = []
    args_list = None

    if params.cb:
        in_txt = copy_from_clipboard()
        args_list = in_txt.splitlines()
        print(f"args_list: {to_str(args_list)}")
        # input("press any key")

    while True:
        if params.cb:
            if not args_list:
                break
            args = args_list.pop(0)
        elif params.list and os.path.isfile(params.list):
            args_list = [args.strip() for args in open(params.list, 'r').readlines()]
            args_list = [args for args in args_list if args and args not in args_history]

            if not args_list:
                print(f'completed processing all args in {params.list}')
                input('Press enter to continue\n')
                continue
            else:
                args = args_list[0]
        else:
            args = input('\nEnter args\n')

        cmd = f'{params.cmd} {args}'

        print(f'running: {cmd}')
        # input("press any key")
        os.system(cmd)

        args_history.append(args)

    # input("press any key")


if __name__ == '__main__':
    main()
