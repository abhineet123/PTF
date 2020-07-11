import sys
import os

from pprint import pformat
from datetime import datetime

import paramparse

# from Misc import processArguments

class Struct:
    def __init__(self, entries):
        self.__dict__.update(entries)


def main():
    params = {
        'in_fname': '',
        'start_id': 0,
        'end_id': -1,
        'server': '',
        'pane_id': '2.0',
        'pane_id_sep': '>',
    }
    paramparse.process_dict(params)

    # processArguments(sys.argv[1:], params)

    _in_fname = params['in_fname']
    start_id = params['start_id']
    end_id = params['end_id']
    server = params['server']
    pane_id_sep = params['pane_id_sep']
    pane_id = params['pane_id']

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
            if pane_id_sep in in_fname:
                _pane_id, _line = in_fname.split(pane_id_sep)
            else:
                _pane_id = pane_id
                _line = in_fname
            lines = ['## @{}.{}'.format(server, _pane_id), _line]
            in_fnames = [in_fname, ]

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

                print('\nReading from: {}'.format(in_fname_path))
                lines = open(in_fname_path, 'r').readlines()

            # print('lines: {}'.format(pformat(lines)))

            pane_to_commands = {}
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
                            print('skipping {} with too small cmd_id'.format(pane_id))
                            continue

                        if cmd_id > end_id > start_id:
                            print('skipping {} with too large cmd_id'.format(pane_id))
                            break

                        pane_to_commands[pane_id] = 'tmux send-keys -t {}'.format(pane_id)
                    continue
                elif _line.startswith('# '):
                    continue

                if server and pane_id and not pane_id.startswith(server):
                    # print('skipping {} with invalid server'.format(pane_id))
                    if pane_id in pane_to_commands:
                        del pane_to_commands[pane_id]
                    continue

                time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")

                pane_to_commands[pane_id] = '{} "{} @ time_stamp={} 2>&1 | tee {}.log" Enter Enter'.format(
                    pane_to_commands[pane_id], _line, time_stamp, time_stamp)

            # print('pane_to_commands: {}'.format(pformat(pane_to_commands)))

            for pane_id in pane_to_commands:
                print('running command in {}'.format(pane_id))
                # print('running: {}'.format(pane_to_commands[pane_id]))
                # esc_command = 'tmux send-keys -t {} Escape'.format(pane_id)
                # os.system(esc_command)
                os.system(pane_to_commands[pane_id])

                # cmd_prefix = 'tmux send-keys -t {}'.format(pane_id)
                # for _line in pane_to_commands[pane_id]:
                #     cmd = '{} "{}" Enter'.format(cmd_prefix, _line)
                #     print('running: {}'.format(cmd))
                #     os.system(cmd)


if __name__ == '__main__':
    main()
