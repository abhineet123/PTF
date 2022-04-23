import os
import difflib
import shutil
import subprocess

import paramparse


class Params:
    def __init__(self):
        self.src = ''
        self.dst = ''


# params = Params()
params = paramparse.process(Params)  # type: Params

assert os.path.exists(params.src), "src does not exist: {}".format(params.src)

f1 = params.dst
cmp = params.dst + '.cmp'

if os.path.exists(params.dst):
    f2 = params.dst + '.backup'
    shutil.move(f1, f2)
else:
    f2 = None

tree_cmd = 'tree {} /a /f > {}'.format(params.src, params.dst)
os.system(tree_cmd)

if f2 is not None:

    file1 = open(f1, 'r',
                 # encoding="utf-8"
                 ).readlines()
    file2 = open(f2, 'r',
                 # encoding="utf-8"
                 ).readlines()

    diffs = difflib.unified_diff(file2, file1, n=0)

    # htmlDiffer = difflib.HtmlDiff()
    # htmldiffs = htmlDiffer.make_file(file1, file2)

    diffs = list(diffs)

    vol_serial_number_lines = [k for k in diffs
                               if k.startswith('-Volume serial number') or
                               k.startswith('+Volume serial number')]
    if vol_serial_number_lines:
        vol_serial_number_idx = max(diffs.index(k) for k in vol_serial_number_lines)
        if vol_serial_number_idx == len(diffs) - 1:
            diffs = []
        else:
            diffs = diffs[vol_serial_number_idx + 1:]

    if diffs:
        with open(cmp, 'w',
                  # encoding="utf-8"
                  ) as outfile:
            # outfile.write('{}\t{}\n'.format(f2, f1))
            for diff in diffs:
                outfile.write(diff)

        # subprocess.call("start " + cmp, shell=True)
        os.startfile(cmp)
