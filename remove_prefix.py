import sys
import os, stat
import random
from Misc import sortKey, processArguments
import inspect

import ctypes


def is_hidden(filepath):
    name = os.path.basename(os.path.abspath(filepath))
    return name.startswith('.') or has_hidden_attribute(filepath)


def has_hidden_attribute2(filepath):
    return bool(os.stat(filepath).st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)


def has_hidden_attribute(filepath):
    # print('filepath: {}'.format(filepath))
    try:
        attrs = ctypes.windll.kernel32.GetFileAttributesW(unicode(filepath))
        assert attrs != -1
        result = bool(attrs & 2)
    except (UnicodeDecodeError,):
        result = False
    except (AttributeError, AssertionError):
        result = False
    return result


def main():
    params = {
        'seq_prefix': ['Seq', ],
        'seq_prefix_filter': '',
        'seq_prefix_ext': '',
        'seq_root_dir': '.',
        'seq_start_id': -1,
        'shuffle_files': 0,
        'filename_fmt': 0,
        'write_log': 1,
        'target_ext': '.jpg',
        'prefix_len': 11,
        'recursive': 0,
        'recursive_src': 1,
        'suffix_mode': 0,
    }
    processArguments(sys.argv[1:], params)
    seq_prefix = params['seq_prefix']
    seq_prefix_filter = params['seq_prefix_filter']
    seq_prefix_ext = params['seq_prefix_ext']
    _seq_root_dir = params['seq_root_dir']
    seq_start_id = params['seq_start_id']
    shuffle_files = params['shuffle_files']
    filename_fmt = params['filename_fmt']
    write_log = params['write_log']
    target_ext = params['target_ext']
    prefix_len = params['prefix_len']
    recursive = params['recursive']
    recursive_src = params['recursive_src']
    suffix_mode = params['suffix_mode']

    excluded_files = ['rseq_log.txt', 'Thumbs.db']

    print('seq_prefix: {}'.format(seq_prefix))

    if len(seq_prefix) == 1:
        seq_prefix = seq_prefix[0]
    elif len(seq_prefix) == 2:
        seq_prefix, seq_prefix_filter = seq_prefix
    elif len(seq_prefix) == 3:
        seq_prefix, seq_prefix_filter, seq_prefix_ext = seq_prefix

    print('seq_prefix: {}'.format(seq_prefix))
    print('seq_prefix_filter: {}'.format(seq_prefix_filter))
    print('seq_prefix_ext: {}'.format(seq_prefix_ext))

    script_filename = inspect.getframeinfo(inspect.currentframe()).filename
    script_path = os.path.dirname(os.path.abspath(script_filename))

    prefix_file_paths = None

    if target_ext:
        print('target_ext: {:s}'.format(target_ext))

    _seq_root_dir = os.path.abspath(_seq_root_dir)

    if os.path.isdir(_seq_root_dir):
        seq_root_dirs = [_seq_root_dir]
    elif os.path.isfile(_seq_root_dir):
        seq_root_dirs = [x.strip() for x in open(_seq_root_dir).readlines() if x.strip()]
    else:
        raise IOError('Invalid seq_root_dir: {}'.format(_seq_root_dir))

    if write_log:
        log_dir = os.path.join(script_path, 'log')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, 'rmpf_log.txt')
        print(('Saving log to {}'.format(log_file)))
        log_fid = open(log_file, 'w')

    all_src_file_names = []
    all_src_file_names_flat = []
    for seq_root_dir in seq_root_dirs:
        seq_root_dir = os.path.abspath(seq_root_dir)

        print('Processing: {}'.format(seq_root_dir))
        if recursive_src:
            src_file_gen = [[os.path.join(dirpath, f) for f in filenames if f not in excluded_files]
                            for (dirpath, dirnames, filenames) in os.walk(seq_root_dir, followlinks=False)]
            src_file_names = [item for sublist in src_file_gen for item in sublist]

        else:
            src_file_names = [os.path.join(seq_root_dir, f) for f in os.listdir(seq_root_dir)
                              if os.path.isfile(os.path.join(seq_root_dir, f))
                              and f not in excluded_files]
        if shuffle_files:
            print('Shuffling files...')
            random.shuffle(src_file_names)
        else:
            src_file_names.sort(key=sortKey)

        all_src_file_names.append(src_file_names)
        all_src_file_names_flat += src_file_names

    for src_file_names in all_src_file_names:
        file_count = 1
        n_files = len(src_file_names)

        for src_file_path in src_file_names:
            src_dir = os.path.dirname(src_file_path)
            src_fname = os.path.basename(src_file_path)

            src_fname_no_ext, src_ext = os.path.splitext(src_fname)

            if target_ext and src_ext != target_ext:
                print(('Ignoring file {} with invalid extension {}'.format(src_fname, src_ext)))
                continue

            # src_path = os.path.join(seq_root_dir, src_fname)
            src_path = src_file_path

            if is_hidden(src_path):
                print('Skipping hidden file: {}'.format(src_path))
                continue
            if suffix_mode:
                dst_fname_noext = src_fname_no_ext[:-prefix_len]
            else:
                dst_fname_noext = src_fname_no_ext[prefix_len:]

            # dst_fname_noext, dst_fname_ext = os.path.splitext(dst_fname)

            dst_fname = dst_fname_noext + src_ext

            dst_path = os.path.join(src_dir, dst_fname)
            if src_path != dst_path:

                seq_id = 0
                while os.path.exists(dst_path):
                    seq_id += 1
                    dst_fname = '{:s}_{:06d}{:s}'.format(dst_fname_noext, seq_id, src_ext)
                    dst_path = os.path.join(src_dir, dst_fname)

                try:
                    os.rename(src_path, dst_path)
                except WindowsError as e:
                    print('Renaming of {} failed: {}'.format(src_path, e))

            print('{} --> {}\n'.format(src_fname, dst_fname))
            if write_log:
                log_fid.write('{}\t{}\n'.format(src_fname, dst_fname))

            if file_count % 10 == 0 or file_count == n_files:
                print('Done {:d}/{:d}'.format(file_count, n_files))

            file_count += 1
        if write_log:
            log_fid.close()


if __name__ == '__main__':
    main()
