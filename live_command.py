import os
import sys
import time

import paramparse
from Misc import sortKey


class Params:
    def __init__(self):
        self.cfg = ()
        self.cmd = ''
        self.list = ''


def main():
    params: Params = paramparse.process(Params)
    assert params.cmd, "params must be provided"

    args_history = []
    while True:
        if params.list and os.path.isfile(params.list):
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
        os.system(cmd)

        args_history.append(args)

if __name__ == '__main__':
    main()
