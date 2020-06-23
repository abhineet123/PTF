import sys
import os
from pprint import pprint
from Misc import processArguments

params = {
    'in_fname': 'all.txt',
    'start_id': 0,
    'end_id': -1,
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)

in_fname = params['in_fname']
start_id = params['start_id']
end_id = params['end_id']

lines = open(in_fname, 'r').readlines()
pane_to_commands = {}
# pprint(lines)
cmd_id = 0
for line in lines:
    _line = line.strip()
    if not _line:
        continue
    if _line.startswith('@ '):
        pane_id = int(_line.replace('@', ''))
        if pane_id not in pane_to_commands:
            cmd_id += 1
            if cmd_id < start_id:
                continue
            if cmd_id > end_id > start_id:
                break
            pane_to_commands[pane_id] = 'tmux send-keys -t {}'.format(pane_id)
        continue
    elif _line.startswith('# '):
        continue
    pane_to_commands[pane_id] = '{} "{}" Enter'.format(pane_to_commands[pane_id], _line)

for pane_id in pane_to_commands:
    # print('running: {}'.format(pane_to_commands[pane_id]))
    os.system(pane_to_commands[pane_id])

    # cmd_prefix = 'tmux send-keys -t {}'.format(pane_id)
    # for _line in pane_to_commands[pane_id]:
    #     cmd = '{} "{}" Enter'.format(cmd_prefix, _line)
    #     print('running: {}'.format(cmd))
    #     os.system(cmd)
