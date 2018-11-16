__author__ = 'Tommy'
import os

root_dir = 'C:/Videos'
in_fname = '#songs new.m3u'
out_fname = '#songs new corr.m3u'

in_path = '{:s}/{:s}'.format(root_dir, in_fname)
out_path = '{:s}/{:s}'.format(root_dir, out_fname)

out_file = open(out_path, 'w')
for line in open(in_path, 'r').readlines():
    line = line.rstrip()
    file_path = '{:s}/{:s}'.format(root_dir, line)
    if os.path.isfile(file_path):
        out_file.write('{:s}\n'.format(line))
    else:
        file_parts=line.split('\\')
        dir=file_parts[0]
        fname=file_parts[-1]
        out_file.write('{:s}/{:s}-{:s}\n'.format(dir, dir, fname))
