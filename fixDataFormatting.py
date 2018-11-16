import datetime
from dateutil.parser import parse

in_filename = 'list.txt'
out_filename = 'list_out.txt'

# in_fmt = None
in_fmt = '%d/%m/%Y'

# out_fmt = '%Y/%m/%d'
out_fmt = '%Y-%m-%d'

in_lines = open(in_filename).readlines()
out_file = open(out_filename, 'w')

for line in in_lines:
    in_line = line.strip()
    if in_fmt is None:
        out_line = parse(in_line).strftime(out_fmt)
    else:
        out_line = datetime.datetime.strptime(in_line, in_fmt).strftime(out_fmt)
    print 'reformatting {:s} to {:s}'.format(in_line, out_line)
    out_file.write('{:s}\n'.format(out_line))
out_file.close()

