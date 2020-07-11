import sys
import os

from pprint import pformat
from datetime import datetime

import paramparse


def write(_str):
    with open('rpit.log', 'a') as fid:
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

    while True:
        lines = None
        if _in_fname:
            in_fname = _in_fname
            _in_fname = ''
        else:
            in_fname = input('\nEnter script path or command\n')

        in_fname_no_ext, in_fname_ext = os.path.splitext(os.path.basename(in_fname))
        if in_fname_ext == '.bsh':
            in_fnames = [in_fname, ]
        elif in_fname_ext == '.bshm':
            in_fnames = open(in_fname, 'r').readlines()
            in_fnames = [__in_fname.strip() for __in_fname in in_fnames if __in_fname.strip()]
        else:
            tokens = in_fname.split(' ')
            try:
                _pane_id = int(tokens[0])
            except ValueError:
                try:
                    _pane_id = float(tokens[0])
                    _pane_id, _line = in_fname.split(pane_id_sep)
                except ValueError:
                    _pane_id = pane_id_default
                    _line = in_fname
                else:
                    _pane_id = str(_pane_id)
                    _line = in_fname[len(tokens[0]) + 1:]
            else:
                _pane_id = '{}.0'.format(_pane_id)
                _line = in_fname[len(tokens[0]) + 1:]

            lines = ['## @ {}:{}'.format(server, _pane_id), _line]
            in_fnames = [in_fname, ]

            write('lines:\n{}'.format(lines))

        src_dir = os.getcwd()
        src_file_gen = [[(f, os.path.join(dirpath, f)) for f in filenames]
                        for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)]
        fname_to_path = dict([item for sublist in src_file_gen for item in sublist])

        for in_fname in in_fnames:
            if lines is None:
                try:
                    in_fname_path = fname_to_path[in_fname]
                except KeyError:
                    raise IOError('invalid file name: {}'.format(in_fname))

                write('\nReading from: {}'.format(in_fname_path))
                lines = open(in_fname_path, 'r').readlines()

            # write('lines: {}'.format(pformat(lines)))

            pane_to_commands = {}
            pane_to_log_paths = {}
            # pprint(lines)
            cmd_id = 0
            pane_id = ''
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

                        pane_to_commands[pane_id] = 'tmux send-keys -t {}'.format(pane_id)
                    continue
                elif _line.startswith('# '):
                    continue

                if server and pane_id and not pane_id.startswith(server):
                    # write('skipping {} with invalid server'.format(pane_id))
                    if pane_id in pane_to_commands:
                        del pane_to_commands[pane_id]
                    continue

                time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")

                if enable_logging:
                    log_path = os.path.join(log_dir, '{}.log'.format(time_stamp))
                    _line = '{} @ time_stamp={} 2>&1 | tee {}'.format(_line, time_stamp, log_path)
                    pane_to_log_paths[pane_id] = log_path

                pane_to_commands[pane_id] = '{} "{}" Enter Enter'.format(pane_to_commands[pane_id], _line)

            # write('pane_to_commands: {}'.format(pformat(pane_to_commands)))

            for pane_id in pane_to_commands:
                txt = 'running command in {}'.format(pane_id)
                # write('running: {}'.format(pane_to_commands[pane_id]))
                # esc_command = 'tmux send-keys -t {} Escape'.format(pane_id)
                # os.system(esc_command)
                if enable_logging:
                    mkdir_cmd = 'mkdir -p {}'.format(log_dir)
                    os.system('tmux send-keys -t {} "{}" Enter'.format(pane_id, mkdir_cmd))
                    txt += ' with logging in {}'.format(pane_to_log_paths[pane_id])

                write(txt)

                os.system(pane_to_commands[pane_id])
                if enable_logging:
                    zip_path = pane_to_log_paths[pane_id].replace('.log', '.zip')
                    zip_cmd = 'zip -rm {} {}'.format(zip_path, pane_to_log_paths[pane_id])
                    os.system('tmux send-keys -t {} "{}" Enter'.format(pane_id, zip_cmd))


if __name__ == '__main__':
    main()
