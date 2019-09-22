import keyboard
import time
import os, sys

from Misc import processArguments, sortKey

if __name__ == '__main__':
    params = {
        'config': 0,
    }
    processArguments(sys.argv[1:], params)
    config = params['config']

    time.sleep(1)

    keyboard.send('t')
    keyboard.send('enter')
    keyboard.send('ctrl+shift+t')

    keyboard.send('f')
    keyboard.send('enter')
    keyboard.send('ctrl+shift+t')
    keys = list('sstg tb') + ['enter', ]
    keys += list('sudo -s') + ['enter', ]
    keys += list("';';';';") + ['enter', ]
    for key in keys:
        keyboard.send(key)

    keyboard.send('ctrl+shift+t')

    keys = list('sstg2') + ['enter', ]
    keys += list('sstz') + ['enter', ]
    keys += list('sudo -s') + ['enter', ]
    keys += list("'''") + ['enter', ]
    for key in keys:
        keyboard.send(key)

    time.sleep(2)

    keyboard.send('shift+left')

    if config == 0:
        keys = list('tmux a -t grs') + ['enter', ]
        for key in keys:
            keyboard.send(key)

        keyboard.send('shift+right')
        keys = list('tmux a -t orca') + ['enter', ]
        for key in keys:
            keyboard.send(key)
    else:
        keys = list('tmux a -t grs2') + ['enter', ]
        for key in keys:
            keyboard.send(key)

        keyboard.send('shift+right')
        keys = list('tmux a -t orca2') + ['enter', ]
        for key in keys:
            keyboard.send(key)
