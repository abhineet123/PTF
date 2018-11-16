import subprocess
import sys
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
print 'curr_path: ', curr_path

src_dir = '.'
list_fname = 'list.txt'
recursive_search = 1
include_folders = 0
replace_existing = 0

arg_id = 1
if len(sys.argv) > arg_id:
    list_fname = sys.argv[arg_id]
    arg_id += 1
if len(sys.argv) > arg_id:
    recursive_search = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    include_folders = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    replace_existing = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    src_dir = sys.argv[arg_id]
    arg_id += 1

list_path = os.path.join(src_dir, list_fname)
lines = [line.rstrip('\n') for line in open(list_path)]

for line in lines:
    src, dst = line.split()
    command = 'python {:s}/rename.py {:s} {:s} {:d} {:d} {:d}'.format(
	curr_path, src, dst, recursive_search, include_folders, replace_existing)
    subprocess.check_call(command, shell=True)
