import sys
import os
import subprocess

from pprint import pformat
from datetime import datetime
import re

import paramparse

from Misc import linux_path


def write(_str):
    with open('rpip.ansi', 'a') as fid:
        fid.write(_str + '\n')
    print(_str)


def main():
    params = {
        'in_fname': "C:/UofA/PhD/Code/deep_mdp/tracking_module/cmd/res_from_gt.md",
        'working_dir': '',
        'start_id': 0,
        'end_id': -1,
        'server': '',
        'pane_id': '12.0',
        'pane_id_sep': '>',
        'log_dir': 'log/rpip',
        'enable_logging': 1,
    }
    paramparse.process_dict(params)

    # processArguments(sys.argv[1:], params)

    in_fname = params['in_fname']
    working_dir = params['working_dir']
    log_dir = params['log_dir']
    enable_logging = params['enable_logging']

    lines = None

    if working_dir:
        os.chdir(working_dir)

    os.makedirs(log_dir, exist_ok=1)

    write('\nprocessing input: {}'.format(in_fname))

    in_fname_no_ext, in_fname_ext = os.path.splitext(os.path.basename(in_fname))
    if in_fname_ext in ('.bsh', '.md'):
        in_fnames = [in_fname, ]
    elif in_fname_ext == '.bshm':
        try:
            # in_fname_path = fname_to_path[in_fname]
            in_fname_path = in_fname
        except KeyError:
            raise IOError('invalid file name: {}'.format(in_fname))

        in_fnames = open(in_fname_path, 'r').readlines()
        in_fnames = [__in_fname.strip() for __in_fname in in_fnames if __in_fname.strip()
                     and not __in_fname.startswith('#')]

        # write('lines:\n{}'.format(lines))
    commands = []

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")

    for in_fname in in_fnames:
        if in_fname is not None:
            try:
                # in_fname_path = fname_to_path[in_fname]
                in_fname_path = in_fname
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
                if enable_logging:
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

    processes = []
    for _cmd_id, _cmd_data in enumerate(commands):
        _cmd, tee_log_id = _cmd_data
        txt = 'running command {}: {}'.format(_cmd_id, _cmd)
        write(txt)
        # subprocess.Popen(_cmd.split(' '))

        if enable_logging:
            out_fname = tee_log_id + '.ansi'
            out_path = linux_path(log_dir, out_fname)
            write('Writing log to {}\n'.format(out_path))
            f = open(out_path, 'w')
            p = subprocess.Popen(_cmd, stdout=f, stderr=f)
        else:
            write('\n')
            f = None
            p = subprocess.Popen(_cmd)

        processes.append((p, f))

    for p, f in processes:
        p.wait()
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()
