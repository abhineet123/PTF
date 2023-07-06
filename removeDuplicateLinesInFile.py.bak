import sys
import os

in_fname = 'filtered.txt'
out_fname = 'filtered_unique.txt'
retain_filtered = 1
arg_id = 1
if len(sys.argv) > arg_id:
    in_fname = sys.argv[arg_id]
    arg_id += 1
if len(sys.argv) > arg_id:
    out_fname = sys.argv[arg_id]
    arg_id += 1


if not os.path.isfile(in_fname):
    print 'Input file {:s} does not exist'.format(in_fname)
    exit(0)

print('Removing duplicate lines in {:s} to {:s}'.format(in_fname, out_fname))
lines_seen = set() # holds lines already seen
outfile = open(out_fname, "w")
for line in open(in_fname, "r"):
    if line not in lines_seen: # not a duplicate
        outfile.write(line)
        lines_seen.add(line)
print('Found {} unique lines'.format(len(lines_seen)))
outfile.close()