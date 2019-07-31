import sys
import os
import random
from Misc import sortKey, processArguments
import inspect

import ctypes
import os

def is_hidden(filepath):
    name = os.path.basename(os.path.abspath(filepath))
    return name.startswith('.') or has_hidden_attribute(filepath)

import os, stat

def has_hidden_attribute2(filepath):
    return bool(os.stat(filepath).st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)

def has_hidden_attribute(filepath):
    try:
        attrs = ctypes.windll.kernel32.GetFileAttributesW(unicode(filepath))
        assert attrs != -1
        result = bool(attrs & 2)
    except (AttributeError, AssertionError):
        result = False
    return result

def main():
    params = {
        'seq_prefix': 'Seq',
        'seq_root_dir': '.',
        'seq_start_id': -1,
        'shuffle_files': 0,
        'filename_fmt': 0,
        'write_log': 1,
        'target_ext': '',
        'recursive': 0,
    }
    processArguments(sys.argv[1:], params)
    seq_prefix = params['seq_prefix']
    _seq_root_dir = params['seq_root_dir']
    seq_start_id = params['seq_start_id']
    shuffle_files = params['shuffle_files']
    filename_fmt = params['filename_fmt']
    write_log = params['write_log']
    target_ext = params['target_ext']
    recursive = params['recursive']

    # seq_prefix = 'Seq'
    # seq_root_dir = '.'
    # seq_start_id = 1
    # shuffle_files = 1
    # filename_fmt = 0
    # write_log = 1
    # target_ext = ''
    #
    # arg_id = 1
    # if len(sys.argv) > arg_id:
    #     seq_prefix = sys.argv[arg_id]
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     seq_start_id = int(sys.argv[arg_id])
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     shuffle_files = int(sys.argv[arg_id])
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     filename_fmt = int(sys.argv[arg_id])
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     _seq_root_dir = sys.argv[arg_id]
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     write_log = int(sys.argv[arg_id])
    #     arg_id += 1
    # if len(sys.argv) > arg_id:
    #     target_ext = sys.argv[arg_id]
    #     arg_id += 1

    script_filename = inspect.getframeinfo(inspect.currentframe()).filename
    script_path = os.path.dirname(os.path.abspath(script_filename))

    if seq_start_id < 0:
        # extract seq_start_id from seq_prefix
        if os.path.isdir(seq_prefix):
            src_dir = os.path.abspath(seq_prefix)
            print(('Looking for sequence prefix in {}'.format(os.path.abspath(src_dir))))
            if recursive:
                print('Searching recursively')
                src_file_gen = [[f for f in filenames if f != 'Thumbs.db']
                                for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)
                                if dirpath !=os.getcwd()]
                src_file_names = [item for sublist in src_file_gen for item in sublist]
            else:
                src_file_names = [f for f in os.listdir(src_dir) if
                                  os.path.isfile(os.path.join(src_dir, f)) and f != 'Thumbs.db']
            src_file_names.sort(key=sortKey)
            # print 'src_file_names: {}'.format(src_file_names)

            seq_prefix = os.path.splitext(src_file_names[-1])[0]
            print(('Found sequence prefix {}'.format(seq_prefix)))
        split_str = seq_prefix.split('_')
        try:
            seq_start_id = int(split_str[-1]) + 1
            seq_prefix = split_str[0]
        except ValueError:
            seq_start_id = 1
        for _str in split_str[1:-1]:
            seq_prefix = '{}_{}'.format(seq_prefix, _str)

    print('seq_prefix: {:s}'.format(seq_prefix))
    print('seq_start_id: {:d}'.format(seq_start_id))
    print('shuffle_files: {:d}'.format(shuffle_files))
    print('file_fmt: {:d}'.format(filename_fmt))

    if target_ext:
        print('target_ext: {:s}'.format(target_ext))

    _seq_root_dir = os.path.abspath(_seq_root_dir)

    if os.path.isdir(_seq_root_dir):
        seq_root_dirs = [_seq_root_dir]
    elif os.path.isfile(_seq_root_dir):
        seq_root_dirs = [x.strip() for x in open(_seq_root_dir).readlines() if x.strip()]
    else:
        raise IOError('Invalid seq_root_dir: {}'.format(_seq_root_dir))

    # if write_log:
    #     log_dir = 'rseq_log'
    #     if not os.path.isdir(log_dir):
    #         os.makedirs(log_dir)
    #     print('Saving log to {}'.format(log_dir))
    if write_log:
        log_dir = os.path.join(script_path, 'log')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, 'rseq_log.txt')
        print(('Saving log to {}'.format(log_file)))
        log_fid = open(log_file, 'w')

    for seq_root_dir in seq_root_dirs:
        print('Processing: {}'.format(seq_root_dir))
        src_file_names = [f for f in os.listdir(seq_root_dir) if os.path.isfile(os.path.join(seq_root_dir, f))
                          and f != 'rseq_log.txt'
                          and f != 'Thumbs.db'
                          ]
        if shuffle_files:
            print('Shuffling files...')
            random.shuffle(src_file_names)
        else:
            src_file_names.sort(key=sortKey)

        seq_id = seq_start_id
        file_count = 1
        n_files = len(src_file_names)

        for src_fname in src_file_names:
            filename, file_extension = os.path.splitext(src_fname)

            if target_ext and file_extension[1:] != target_ext:
                print(('Ignoring file {} with invalid extension {}'.format(src_fname, file_extension)))
                continue

            src_path = os.path.join(seq_root_dir, src_fname)

            if is_hidden(src_path):
                print('Skipping hidden file: {}'.format(src_path))
                continue

            if filename_fmt == 0:
                dst_fname = '{:s}_{:d}{:s}'.format(seq_prefix, seq_id, file_extension)
            else:
                dst_fname = '{:s}{:06d}{:s}'.format(seq_prefix, seq_id, file_extension)
            dst_path = os.path.join(seq_root_dir, dst_fname)

            if src_path != dst_path:
                while os.path.exists(dst_path):
                    seq_id += 1
                    if filename_fmt == 0:
                        dst_fname = '{:s}_{:d}{:s}'.format(seq_prefix, seq_id, file_extension)
                    else:
                        dst_fname = '{:s}{:06d}{:s}'.format(seq_prefix, seq_id, file_extension)
                    dst_path = os.path.join(seq_root_dir, dst_fname)
                try:
                    os.rename(src_path, dst_path)
                except WindowsError as e:
                    print('Renaming of {} failed: {}'.format(src_path, e))
            if write_log:
                log_fid.write('{}\t{}\n'.format(src_fname, dst_fname))
            seq_id += 1
            if file_count % 10 == 0 or file_count == n_files:
                print('Done {:d}/{:d}'.format(file_count, n_files))
            file_count += 1
        if write_log:
            log_fid.close()

if __name__ == '__main__':
    main()