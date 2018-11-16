import fnmatch
import os
import sys
import shutil

root_dir = '.'
dst_dir = 'moved'
src_substr = '4u'

arg_id = 1
if len(sys.argv) > arg_id:
    src_substr = sys.argv[arg_id]
    arg_id += 1
if len(sys.argv) > arg_id:
    dst_dir = sys.argv[arg_id]
    arg_id += 1
if len(sys.argv) > arg_id:
    root_dir = sys.argv[arg_id]
    arg_id += 1

dst_path = '{:s}{:s}{:s}{:s}'.format(root_dir,os.sep, dst_dir, os.sep)
print 'Searching for {:s} in {:s} to move to {:s}'.format(src_substr, root_dir, dst_dir)
if not os.path.exists(dst_path):
    print 'Destination folder does not exist. Creating it...'
    os.mkdir(dst_path)

src_fnames = []
for root, dirnames, filenames in os.walk(root_dir):
    for filename in fnmatch.filter(filenames, '*{:s}*'.format(src_substr)):
        src_fnames.append(os.path.join(root, filename))
print 'Found {:d} matches'.format(len(src_fnames))
for src_fname in src_fnames:
    dst_fname = src_fname.replace('{:s}{:s}'.format(root_dir, os.sep), '{:s}{:s}{:s}{:s}'.format(
        root_dir,os.sep, dst_dir, os.sep))
    splitlocaldir = dst_fname.split(os.sep)
    splitlocaldir.remove(splitlocaldir[-1:][0])
    localdir = ""
    for item in splitlocaldir:
        localdir += item + os.sep
    if not os.path.exists(localdir):
        os.makedirs(localdir)
    # dst_fname = '{:s}\{:s}'.format(root_dir, src_fname)
    print 'moving {:s} to {:s}'.format(src_fname, dst_fname)
    # shutil.move(src_fname, dst_fname)
    os.rename(src_fname, dst_fname)
    # print matches