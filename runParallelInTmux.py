import sys
import os
from pprint import pprint
from Misc import processArguments

params = {
    'in_fname': 'all.txt',
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)

in_fname = params['in_fname']

lines = open(in_fname, 'r').readlines()
pane_id = 0
pane_to_commands = {
    pane_id: []
}
pprint(lines)
for line in lines:
    _line = line.strip()
    if not _line:
        continue
    if _line.startswith('# @'):
        pane_id = int(_line.replace('# @', ''))
        if pane_id not in pane_to_commands:
            pane_to_commands[pane_id] = []
        continue
    pane_to_commands[pane_id].append(_line)

for pane_id in pane_to_commands:
    cmd_prefix = 'tmux send-keys -t {}'.format(pane_id)
    for _line in pane_to_commands[pane_id]:
        cmd = '{} "{}" Enter'.format(cmd_prefix, _line)
        print('running: {}'.format(cmd))
        os.system(cmd)
