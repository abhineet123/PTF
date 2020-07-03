import sys
import os

import paramparse
from pprint import pformat

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
    }
    paramparse.process_dict(params)

    # processArguments(sys.argv[1:], params)

    _in_fname = params['in_fname']
    start_id = params['start_id']
    end_id = params['end_id']
    server = params['server']

    while True:
        if _in_fname:
            in_fname = _in_fname
            _in_fname = ''
        else:
            in_fname = input('\nEnter script path\n')

        in_fname_no_ext, in_fname_ext = os.path.splitext(os.path.basename(in_fname))
        if in_fname_ext == '.bshm':
            in_fnames = open(in_fname, 'r').readlines()
            in_fnames = [__in_fname.strip() for __in_fname in in_fnames if __in_fname.strip()]
        else:
            in_fnames = [in_fname, ]

        for in_fname in in_fnames:
            print('\nReading from: {}'.format(in_fname))
            lines = open(in_fname, 'r').readlines()

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

                pane_to_commands[pane_id] = '{} Escape Enter "{}" Enter'.format(pane_to_commands[pane_id], _line)

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
