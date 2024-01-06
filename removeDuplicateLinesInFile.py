import os
import paramparse


class Params:
    def __init__(self):
        self.in_fname = ''
        self.out_fname = ''
        self.check_exist = 0


def main():
    params = Params()
    paramparse.process(params, verbose=1)

    out_fname = params.out_fname

    assert params.in_fname, "in_fname must be provided"
    assert os.path.isfile(params.in_fname), f'Input file {params.in_fname:s} does not exist'

    if not out_fname:
        out_fname = f'{params.in_fname}.unique'

    print(f'Removing duplicate lines in {params.in_fname:s} to {out_fname:s}')

    lines = open(params.in_fname, "r").readlines()

    print(f'Found {len(lines)} unique lines')

    if params.check_exist:
        lines = [line for line in lines if os.path.exists(line.strip())]
    print(f'Found {len(lines)} unique and existent lines')

    open(out_fname, "w").write(''.join(lines))


if __name__ == '__main__':
    main()
