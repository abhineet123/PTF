import sys
import os

from pprint import pformat
from datetime import datetime
import re

import paramparse


def write(_str):
    with open('rpit.ansi', 'a') as fid:
        fid.write(_str + '\n')
    print(_str)


def main():
    params = {
        'in_fname': '',
        'start_id': 0,
        'end_id': -1,
        'server': '',
        'pane_id': '12.0',
        'pane_id_sep': '>',
        'log_dir': 'log/tee',
        'enable_logging': 0,
    }
    paramparse.process_dict(params)

    # processArguments(sys.argv[1:], params)

    _in_fname = params['in_fname']
    start_id = params['start_id']
    end_id = params['end_id']
    server = params['server']
    pane_id_sep = params['pane_id_sep']
    pane_id_default = params['pane_id']
    log_dir = params['log_dir']
    enable_logging = params['enable_logging']

    prev_in_fname = None
    while True:
        lines = None
        if _in_fname:
            in_fname = _in_fname
            _in_fname = ''
        else:
            in_fname = input('\nEnter script path or command\n')

        if not in_fname.strip():
            if prev_in_fname is not None:
                in_fname = prev_in_fname
            else:
                continue

        write('\nprocessing input: {}'.format(in_fname))

        src_dir = os.getcwd()
        src_file_gen = [[(f, os.path.join(dirpath, f)) for f in filenames]
                        for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)]
        fname_to_path = dict([item for sublist in src_file_gen for item in sublist])

        prev_in_fname = in_fname

        in_fname_no_ext, in_fname_ext = os.path.splitext(os.path.basename(in_fname))
        if in_fname_ext == '.bsh':
            in_fnames = [in_fname, ]
        elif in_fname_ext == '.bshm':
            try:
                in_fname_path = fname_to_path[in_fname]
            except KeyError:
                raise IOError('invalid file name: {}'.format(in_fname))

            in_fnames = open(in_fname_path, 'r').readlines()
            in_fnames = [__in_fname.strip() for __in_fname in in_fnames if __in_fname.strip()
                         and not __in_fname.startswith('#')]
        else:
            tokens = in_fname.split(' ')
            try:
                _pane_id = int(tokens[0])
            except ValueError:
                try:
                    _pane_id = float(tokens[0])
                except ValueError as e:
                    print('float pane id failed: {}'.format(e))
                    _pane_id = pane_id_default
                    _line = in_fname
                else:
                    _pane_id = tokens[0]
                    _line = in_fname[len(tokens[0]) + 1:]
            else:
                _pane_id = '{}.0'.format(_pane_id)
                _line = in_fname[len(tokens[0]) + 1:]

            lines = ['## @ {}:{}'.format(server, _pane_id), _line]
            in_fnames = [None, ]

            # write('lines:\n{}'.format(lines))

        all_pane_ids = []

        for in_fname in in_fnames:
            if in_fname is not None:
                try:
                    in_fname_path = fname_to_path[in_fname]
                except KeyError:
                    raise IOError('invalid file name: {}'.format(in_fname))

                write('\nReading from: {}'.format(in_fname_path))
                lines = open(in_fname_path, 'r').readlines()
            else:
                assert lines is not None, "Both lines and in_fname cannot be None"

            # write('lines: {}'.format(pformat(lines)))

            pane_to_commands = {}
            pane_to_log = {}
            # pprint(lines)
            cmd_id = 0
            pane_id = ''

            command_lines = [_line for _line in lines if not _line.startswith('## @ ') and not _line.startswith('# ')]

            for line in lines:
                _line = line.strip()
                if not _line:
                    continue

                if _line.startswith('## @ '):
                    pane_id = _line.replace('## @ ', '')

                    if pane_id not in pane_to_commands:
                        cmd_id += 1

                        if cmd_id < start_id:
                            write('skipping {} with too small cmd_id'.format(pane_id))
                            continue

                        if cmd_id > end_id > start_id:
                            write('skipping {} with too large cmd_id'.format(pane_id))
                            break

                        pane_to_commands[pane_id] = []
                        pane_to_log[pane_id] = []

                    # pane_to_commands[pane_id].append('tmux send-keys -t {}'.format(pane_id))
                    continue
                elif _line.startswith('# '):
                    continue

                if server and pane_id and not pane_id.startswith(server):
                    # write('skipping {} with invalid server'.format(pane_id))
                    if pane_id in pane_to_commands:
                        del pane_to_commands[pane_id]
                        del pane_to_log[pane_id]
                    continue

                if enable_logging:
                    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
                    log_fname = '{}.ansi'.format(time_stamp)
                    log_path = os.path.join(log_dir, log_fname)
                    tee_log_id = '{}:{}'.format(pane_id, time_stamp)
                    _line = '{} @ tee_log={} 2>&1 | tee {}'.format(_line, tee_log_id, log_path)

                    """disable python output buffering to ensure in-order output in the logging fine"""
                    if _line.startswith('python '):
                        _line = _line.replace('python ', 'python -u ', 1)
                    elif _line.startswith('python3 '):
                        _line = _line.replace('python3 ', 'python3 -u ', 1)
                    elif _line.startswith('python2 '):
                        _line = _line.replace('python2 ', 'python2 -u ', 1)
                    pane_to_log[pane_id].append(log_fname)

                list_start_indices = [m.start() for m in re.finditer(_line, '{')]
                list_end_indices = [m.start() for m in re.finditer(_line, '}')]

                assert len(list_start_indices) == len(list_end_indices), \
                    "mismatch between number of list start and end markers"

                _multi_token_lines = [_line, ]
                for _start_id, _end_id in zip(list_start_indices, list_end_indices):
                    substr = _line[_start_id:_end_id + 1]

                    replacement_vals = paramparse.str_to_tuple(substr[1:-1])

                    temp_line_list = []
                    for __line in _multi_token_lines:
                        for _val in replacement_vals:
                            new_line = __line.replace(substr, _val, count=1)
                            temp_line_list.append(new_line)

                    _multi_token_lines = temp_line_list

                # _line_tokens = _line.split(' ')
                # _multi_tokens = [(__id, token) for __id, token in enumerate(_line_tokens)
                #                  if token.startswith('__rpit_multi__')]
                # if _multi_tokens:
                #     assert len(_multi_tokens) == 1, "only singluar multi_tokens per line supported for now"
                #     __id, _multi_token = _multi_tokens[0]
                #
                #     _multi_token_lines = []
                #     _arg_name, _file_name, _start_id, _end_id = _multi_token.split(':')
                #     _multi_token_arg_vals = open(_file_name, 'r').readlines()[_start_id:_end_id + 1]
                #
                #     for _arg_val in _multi_token_arg_vals:
                #         _multi_tokens_copy = _multi_tokens[:]
                #         _multi_tokens_copy[__id] = '{}={}'.format(_arg_name, _arg_val)
                #
                #         _multi_token_line = ' '.join(_multi_tokens_copy)
                #         _multi_token_lines.append(_multi_token_line)
                # else:
                #     _multi_token_lines = [_line, ]

                for __line in _multi_token_lines:
                    pane_to_commands[pane_id].append('tmux send-keys -t {} "{}" Enter Enter'.format(
                        pane_id,
                        # pane_to_commands[pane_id][-1],
                        __line)
                    )

            # write('pane_to_commands: {}'.format(pformat(pane_to_commands)))
            lines = None

            for pane_id in pane_to_commands:
                for _cmd_id, _cmd in enumerate(pane_to_commands[pane_id]):
                    txt = 'running command {} in {}'.format(_cmd_id, pane_id)
                    if enable_logging:
                        mkdir_cmd = 'mkdir -p {}'.format(log_dir)
                        # os.system('tmux send-keys -t {} "{}" Enter'.format(pane_id, mkdir_cmd))

                    # os.system(_cmd)

                    if enable_logging:
                        log_fname = pane_to_log[pane_id][_cmd_id]
                        zip_fname = log_fname.replace('.ansi', '.zip')
                        zip_path = os.path.join(log_dir, zip_fname)

                        zip_cmd = '(cd {} && zip -rm {} {})'.format(log_dir, zip_fname, log_fname)
                        # os.system('tmux send-keys -t {} "{}" Enter'.format(pane_id, zip_cmd))
                        txt += ' with logging in {}'.format(zip_path)

                    write(txt)

            all_pane_ids += list(pane_to_commands.keys())

        all_pane_ids_str = '__'.join(all_pane_ids).replace(':', '_')
        write('{}'.format(all_pane_ids_str))


if __name__ == '__main__':
    main()
