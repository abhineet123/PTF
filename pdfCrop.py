import os
import sys
import time

import paramparse
from Misc import processArguments, sortKey


class Params:
    def __init__(self):
        self.cfg = ()
        self.file_ext = 'pdf'
        self.folder_name = '.'
        self.live = 0
        self.recursive = 0
        self.oxps = 0


def run(params: Params, src_files=None, last_run_time=None):
    if params.oxps:
        params.file_ext = 'oxps'

    if src_files is None or not src_files[0]:
        if params.recursive:
            src_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                             f.endswith(params.file_ext)]
                            for (dirpath, dirnames, filenames) in os.walk(params.folder_name, followlinks=True)]
            src_files = [item for sublist in src_file_gen for item in sublist]
        else:
            src_files = [os.path.join(params.folder_name, f) for f in os.listdir(params.folder_name)
                         if os.path.isfile(os.path.join(params.folder_name, f)) and f.endswith(params.file_ext)]

        if not params.oxps:
            src_files = [k for k in src_files if not k.endswith('-crop.pdf')]

        if last_run_time is not None:
            # mtimes = {os.path.basename(k): os.path.getmtime(k) for k in src_files if os.path.getmtime(k)}
            # print(f'last_run_time: {last_run_time}')
            # print(f'mtimes: {mtimes}')
            src_files = [k for k in src_files if os.path.getmtime(k) > last_run_time]

        src_files.sort(key=sortKey)

    if not src_files:
        print('\nNo new files found\n')
        return

    n_files = len(src_files)

    for i, _f in enumerate(src_files):
        __f = os.path.splitext(_f)[0]
        if params.oxps:
            cmd = f'gxpswin64 -sDEVICE=pdfwrite -sOutputFile={__f}.pdf -dNOPAUSE {__f}.oxps'
            print(f'{i + 1}/{n_files} {cmd}')
            os.system(cmd)

        cmd = f'pdfcrop {__f}.pdf'
        print(f'{i + 1}/{n_files} {cmd}')
        os.system(cmd)


def main():
    params: Params = paramparse.process(Params)
    if not params.live:
        run(params)

    last_run_time = None
    while True:
        k = input('\nEnter file name. Leave blank for all new files.\n')
        try:
            run(params, src_files=[k], last_run_time=last_run_time)
        except BaseException as e:
            print('pdfcrop failed:')
            print(e)
        last_run_time = time.time()


if __name__ == '__main__':
    main()
