__author__ = 'Tommy'
import sys
from shutil import copyfile

in_fname = 'sigma.cfg'
root_dir = '.'
separator = ':'
comment_marker = '#'
reset_marker = '-'
out_to_in = 1
arg_id = 1
if len(sys.argv) > arg_id:
    out_to_in = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    in_fname = sys.argv[arg_id]
    arg_id += 1
if len(sys.argv) > arg_id:
    separator = sys.argv[arg_id][0]
    arg_id += 1
if len(sys.argv) > arg_id:
    comment_marker = sys.argv[arg_id][0]
    arg_id += 1
if len(sys.argv) > arg_id:
    reset_marker = sys.argv[arg_id][0]
    arg_id += 1
if len(sys.argv) > arg_id:
    root_dir = sys.argv[arg_id]
    arg_id += 1
if out_to_in:
    out_fname = '{:s}'.format(in_fname)
    copyfile(in_fname, '{:s}.back'.format(in_fname))
else:
    out_fname = '{:s}.out'.format(in_fname)
lines = open('{:s}/{:s}'.format(root_dir, in_fname), 'r').readlines()
out_file = open('{:s}/{:s}'.format(root_dir, out_fname), 'w')
line_id = 0
for line in lines:
    if line[0] == comment_marker:
        out_file.write(line)
        if line[1] == reset_marker:
            line_id = 0
        continue
    while len(line) and line[0] != separator:
        line = line[1:]
    out_file.write('{:02d}{:s}'.format(line_id, line))
    line_id += 1
out_file.close()


