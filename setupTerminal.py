import keyboard
import time

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
time.sleep(2)
keys += list('tmux a') + ['enter', ]
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
keys = list('tmux a') + ['enter', ]
for key in keys:
    keyboard.send(key)
