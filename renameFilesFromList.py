import os
import sys

src_names_fname = 'src_list.txt'
dst_names_fname = 'dst_list.txt'
src_root_dir = '.'
dst_root_dir = '.'
invert_list = 0

arg_id = 1
if len(sys.argv) > arg_id:
    invert_list = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    src_names_fname = sys.argv[arg_id]
    arg_id += 1
if len(sys.argv) > arg_id:
    dst_names_fname = sys.argv[arg_id]
    arg_id += 1
if len(sys.argv) > arg_id:
    src_root_dir = sys.argv[arg_id]
    arg_id += 1
if len(sys.argv) > arg_id:
    dst_root_dir = sys.argv[arg_id]
    arg_id += 1


if not os.path.isfile(dst_names_fname):
    raise SyntaxError('File containing the source file list not found')

if not os.path.exists(src_root_dir):
    raise SyntaxError('Folder containing the source files {:s} does not exist'.format(src_root_dir))

if not os.path.exists(dst_root_dir):
    os.mkdir(dst_root_dir)

src_data_file = open(src_names_fname, 'r')
src_lines = src_data_file.readlines()
src_data_file.close()

dst_data_file = open(dst_names_fname, 'r')
dst_lines = dst_data_file.readlines()
dst_data_file.close()

if invert_list:
    dst_lines = dst_lines[::-1]

n_files = len(src_lines)

if n_files != len(dst_lines):
    raise SyntaxError('No. of lines in the source file list {:d} does not match that in the destination one {:d}'.format(
        n_files, len(dst_lines)))

for file_id in xrange(n_files):
    src_file = src_lines[file_id].strip()
    src_fname, src_ext = os.path.splitext(src_file)
    src_path = unicode('{:s}/{:s}'.format(src_root_dir, src_file), 'utf-8-sig')

    if not os.path.isfile(src_path) and not os.path.isdir(src_path):
        raise SyntaxError('Original file/folder {:s} does not exist'.format(src_path))

    dst_file = dst_lines[file_id].strip()
    dst_fname, dst_ext = os.path.splitext(dst_file)

    # print 'dst_fname: {:s}, dst_ext: {:s}'.format(dst_fname, dst_ext)

    # if src_ext and not dst_ext:
    #     # use the source extension only if the destination extension does not exist
    #     #  and the source extension does exist
    #     dst_path = '{:s}{:s}'.format(dst_root_dir, dst_fname, src_ext)
    # else:
    #     dst_path = '{:s}/{:s}'.format(dst_root_dir, dst_file)

    # print 'Renaming {:s} to {:s}'.format(src_path, dst_file)
    os.rename(src_path , unicode(dst_file + src_ext, 'utf-8-sig'))


