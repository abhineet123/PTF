import sys
import os
from Misc import processArguments

params = {
    'filter_strings': ['magnet'],
    'in_fname': 'all.txt',
    'out_fname': 'filtered.txt',
    # 0: only at start
    # 1: anywhere
    'filter_type': 0,
    'retain_filtered': 1,
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)

filter_strings = params['filter_strings']
filter_type = params['filter_type']
in_fname = params['in_fname']
out_fname = params['out_fname']
retain_filtered = params['retain_filtered']

if not os.path.isfile(in_fname):
    print('Input file {:s} does not exist'.format(in_fname))
    exit(0)

if filter_type == 0:
    print('Filtering lines starting with {} in {:s} to {:s}'.format(filter_strings, in_fname, out_fname))
else:
    print('Filtering lines containing {} in {:s} to {:s}'.format(filter_strings, in_fname, out_fname))
if not retain_filtered:
    print('Removing the filtered lines')

out_fid = open(out_fname, 'w')
lines = open(in_fname, 'r').readlines()
n_filtered_lines = 0
for line in lines:
    retain_line = False
    if filter_type == 0:
        retain_line = any([line.startswith(k) for k in filter_strings])
    elif filter_type == 1:
        retain_line = any([k in line for k in filter_strings])
    if not retain_filtered:
        retain_line = not retain_line
    if retain_line:
        out_fid.write(line)
        n_filtered_lines += 1
print('Done filtering {:d} lines'.format(n_filtered_lines))
out_fid.close()
