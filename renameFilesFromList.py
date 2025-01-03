import os
import shutil
import sys

import paramparse

from Misc import linux_path


class Params:
    src_names_fname = 'list.txt'
    # dst_names_fname = 'dst_list.txt'
    src_root_dir = '.'
    dst_root_dir = '.'
    dst_first = 0
    invert_list = 0

def main():
    params = Params()
    paramparse.process(params)

    src_names_fname = params.src_names_fname
    src_root_dir = params.src_root_dir
    dst_root_dir = params.dst_root_dir
    invert_list = params.invert_list
    dst_first = params.dst_first

    # if not os.path.isfile(dst_names_fname):
    #     raise SyntaxError('File containing the source file list not found')

    if not os.path.exists(src_root_dir):
        raise SyntaxError('Folder containing the source files {:s} does not exist'.format(src_root_dir))

    if not os.path.exists(dst_root_dir):
        os.mkdir(dst_root_dir)

    in_lines = open(src_names_fname, 'r').readlines()

    if '\t' in in_lines[0]:
        src_lines = [in_line.split('\t') for in_line in in_lines]
        src_lines, dst_lines = zip(*src_lines)
        if dst_first:
            src_lines, dst_lines = dst_lines, src_lines
    else:
        dst_lines = in_lines
        src_lines = sorted(k for k in os.listdir(src_root_dir) if k != src_names_fname)

    # dst_data_file = open(dst_names_fname, 'r')
    # dst_lines = dst_data_file.readlines()
    # dst_data_file.close()

    if invert_list:
        dst_lines = dst_lines[::-1]

    n_files = len(src_lines)

    assert n_files == len(
        dst_lines), 'No. of lines in the source file list {:d} does not match that in the destination one {:d}'.format(
        n_files, len(dst_lines))

    for file_id in range(n_files):
        src_file = src_lines[file_id].rstrip()
        src_fname, src_ext = os.path.splitext(src_file)
        src_path = src_file
        # src_path = '{:s}/{:s}'.format(src_root_dir, src_file)

        assert os.path.isfile(src_path) or os.path.isdir(src_path), 'Original file/folder {:s} does not exist'.format(
            src_path)

        dst_file = dst_lines[file_id].rstrip()
        dst_fname, dst_ext = os.path.splitext(dst_file)

        dst_path = dst_file + src_ext

        # print 'dst_fname: {:s}, dst_ext: {:s}'.format(dst_fname, dst_ext)

        # if src_ext and not dst_ext:
        #     # use the source extension only if the destination extension does not exist
        #     #  and the source extension does exist
        #     dst_path = '{:s}{:s}'.format(dst_root_dir, dst_fname, src_ext)
        # else:
        #     dst_path = '{:s}/{:s}'.format(dst_root_dir, dst_file)

        print('{:s} --> {:s}'.format(src_path, dst_file))
        shutil.move(src_path, dst_path)

if __name__ == '__main__':
    main()
