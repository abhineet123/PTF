import sys
import os
import subprocess
import shlex
import time

from pprint import pformat
from datetime import datetime
import re

import paramparse
import numpy as np


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def write(_str):
    with open('rpip.ansi', 'a') as fid:
        fid.write(_str + '\n')
    print(_str)


def main():
    params = {
        # 'in_fname': "C:/UofA/PhD/Code/deep_mdp/tracking_module/cmd/res_from_gt.md",
        'in_fname': "",
        'working_dir': '',
        'start_id': 0,
        'end_id': -1,
        'server': '',
        'pane_id': '12.0',
        'pane_id_sep': '>',
        'log_dir': 'log/rpip',
        'enable_logging': 1,
        'enable_tee': 0,
        'batch_size': 50,
    }
    paramparse.process_dict(params)

    # processArguments(sys.argv[1:], params)

    _in_fname = params['in_fname']
    working_dir = params['working_dir']
    server = params['server']
    log_dir = params['log_dir']
    enable_logging = params['enable_logging']
    enable_tee = params['enable_tee']
    batch_size = params['batch_size']

    if working_dir:
        os.chdir(working_dir)

    os.makedirs(log_dir, exist_ok=1)

    while True:
        src_dir = os.getcwd()
        src_file_gen = [[(f, os.path.join(dirpath, f)) for f in filenames]
                        for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)]
        fname_to_path = dict([item for sublist in src_file_gen for item in sublist])

        lines = None
        if _in_fname:
            in_fname = _in_fname
            _in_fname = ''
        else:
            in_fname = input('\nEnter script path or command\n')

        write('\nprocessing input: {}'.format(in_fname))

        in_fname_no_ext, in_fname_ext = os.path.splitext(os.path.basename(in_fname))
        if in_fname_ext in ('.bsh', '.md'):
            in_fnames = [in_fname, ]
        elif in_fname_ext == '.bshm':
            try:
                in_fname_path = fname_to_path[in_fname]
                # in_fname_path = in_fname
            except KeyError:
                raise IOError('invalid file name: {}'.format(in_fname))

            in_fnames = open(in_fname_path, 'r').readlines()
            in_fnames = [__in_fname.strip() for __in_fname in in_fnames if __in_fname.strip()
                         and not __in_fname.startswith('#')]

            # write('lines:\n{}'.format(lines))
        commands = []

        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

        for in_fname in in_fnames:
            if in_fname is not None:
                try:
                    in_fname_path = fname_to_path[in_fname]
                    # in_fname_path = in_fname
                except KeyError:
                    raise IOError('invalid file name: {}'.format(in_fname))

                write('\nReading from: {}'.format(in_fname_path))
                lines = open(in_fname_path, 'r').readlines()
            else:
                assert lines is not None, "Both lines and in_fname cannot be None"

            basename = os.path.basename(in_fname_path)
            basename_no_ext, _ = os.path.splitext(basename)

            # write('lines: {}'.format(pformat(lines)))

            # command_lines = [_line for _line in lines if not _line.startswith('## @ ') and not _line.startswith('# ')]

            valid_line_id = 0
            for line in lines:
                _line = line.strip()
                if not _line or not _line.startswith('python'):
                    continue

                list_start_indices = [m.start() for m in re.finditer("{", _line)]
                list_end_indices = [m.start() for m in re.finditer("}", _line)]

                assert len(list_start_indices) == len(list_end_indices), \
                    "mismatch between number of list start and end markers"

                _multi_token_lines = [_line, ]
                for _start_id, _end_id in zip(list_start_indices, list_end_indices):
                    substr = _line[_start_id:_end_id + 1]

                    replacement_vals = paramparse.str_to_tuple(substr[1:-1])

                    temp_line_list = []
                    for __line in _multi_token_lines:
                        for _val in replacement_vals:
                            new_line = __line.replace(substr, str(_val), 1)
                            temp_line_list.append(new_line)

                    _multi_token_lines = temp_line_list

                for __line_id, __line in enumerate(_multi_token_lines):
                    tee_log_id = '{}_{}_{}_{}'.format(basename_no_ext, valid_line_id, __line_id, time_stamp)
                    if server:
                        tee_log_id = '{}_{}'.format(tee_log_id, server)

                    if enable_logging:
                        if enable_tee:
                            __line = '{} @ tee_log={}'.format(__line, tee_log_id)

                        """disable python output buffering to ensure in-order output in the logging fine"""
                        if __line.startswith('python '):
                            __line = __line.replace('python ', 'python -u ', 1)
                        elif __line.startswith('python3 '):
                            __line = __line.replace('python3 ', 'python3 -u ', 1)
                        elif __line.startswith('python36 '):
                            __line = __line.replace('python36 ', 'python36 -u ', 1)
                        elif __line.startswith('python2 '):
                            __line = __line.replace('python2 ', 'python2 -u ', 1)

                    commands.append((__line, tee_log_id))
                valid_line_id += 1

        n_commands = len(commands)
        n_batches = int(np.ceil(n_commands / batch_size))
        avg_batch_time = 0

        for batch_id in range(n_batches):

            start_batch_id = int(batch_id * batch_size)
            end_batch_id = min(start_batch_id + batch_size, n_commands)

            actual_batch_size = end_batch_id - start_batch_id

            batch_commands = commands[start_batch_id:end_batch_id]

            batch_start_t = time.time()
            batch_start_time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

            processes = []
            for _cmd_id, _cmd_data in enumerate(batch_commands):
                _cmd, tee_log_id = _cmd_data
                txt = '{}: {}'.format(_cmd_id + start_batch_id, _cmd)
                write(txt)
                # subprocess.Popen(_cmd.split(' '))

                args = shlex.split(_cmd)

                if enable_logging:
                    out_fname = tee_log_id + '.ansi'
                    zip_fname = out_fname.replace('.ansi', '.zip')

                    out_path = linux_path(log_dir, out_fname)
                    zip_path = os.path.join(log_dir, zip_fname)

                    write('{}\n'.format(out_path))
                    write('{}\n'.format(zip_path))

                    f = open(out_path, 'w')
                    p = subprocess.Popen(args, stdout=f, stderr=f)
                else:
                    write('\n')
                    f = out_fname = zip_fname = None
                    p = subprocess.Popen(args)

                processes.append((p, f, out_fname, zip_fname))

            write('{} :: running batch {} / {} with {} commands ...'.format(
                batch_start_time_stamp, batch_id + 1, n_batches, actual_batch_size))

            for p, f, f_name, zip_fname in processes:
                p.wait()

                if f is None:
                    continue

                f.close()

                zip_cmd = 'cd {} && zip -rm {} {}'.format(log_dir, zip_fname, f_name)

                os.system(zip_cmd)
            batch_end_t = time.time()

            batch_time = batch_end_t - batch_start_t
            avg_batch_time += (batch_time - avg_batch_time) / (batch_id + 1)
            batch_end_time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
            write('{} :: Batch {} completed. Time taken: {} sec (avg: {} sec)'.format(
                batch_end_time_stamp, batch_id + 1, batch_time, avg_batch_time))


if __name__ == '__main__':
    main()
