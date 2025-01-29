import os
import sys
import paramparse
from Misc import processArguments, sortKey


class Params:
    def __init__(self):
        self.cfg = ()
        self.file_ext = 'pdf'
        self.folder_name = '.'
        self.live = 0
        self.recursive = 0


def run(params:Params, src_files=None):
    if src_files is None or not src_files[0]:
        if params.recursive:
            src_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                             f.endswith(params.file_ext)]
                            for (dirpath, dirnames, filenames) in os.walk(params.folder_name, followlinks=True)]
            src_files = [item for sublist in src_file_gen for item in sublist]
        else:
            src_files = [os.path.join(params.folder_name, f) for f in os.listdir(params.folder_name)
                         if os.path.isfile(os.path.join(params.folder_name, f)) and f.endswith(params.file_ext)]

        src_files = [k for k in src_files if not k.endswith('-crop.pdf')]

        src_files.sort(key=sortKey)

    n_files = len(src_files)

    for i, _f in enumerate(src_files):
        if not _f.endswith('.pdf'):
            _f = f'{_f}.pdf'
        cmd = f'pdfcrop {_f}'
        print(f'{i + 1}/{n_files} {cmd}')
        os.system(cmd)


def main():
    params: Params = paramparse.process(Params)
    if not params.live:
        run(params)

    while True:
        k = input('Enter file name. Leave blank for all files.\n')
        try:
            run(params, src_files=[k])
        except BaseException as e:
            print('pdfcrop failed:')
            print(e)


if __name__ == '__main__':
    main()
