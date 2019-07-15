import time
import keyboard

def kb_callback(event):
    global _exit

    # print('_params: {}'.format(_params))

    # _type = event.name
    # scan_code = event.scan_code
    # print('scan_code: {}'.format(scan_code))

    _type = event
    print('hotkey: {}'.format(_type))

    if _type == 'ctrl+alt+esc':
        print('removing hotkeys and exiting')
        remove_hotkeys()
        _exit = 1
    # elif _type == 'play/pause media' or _type == -179:
    #     print('sending shift+up')
    #     keyboard.send('shift+up')
    elif _type == 'previous track' or _type == -177:
        print('sending shift+left')
        keyboard.send('shift+left')
    elif _type == 'next track' or _type == -176:
        print('sending shift+right')
        keyboard.send('shift+right')


hotkeys = [
    'ctrl+alt+esc',
    # -179,
    -177,
    -176,
]


def add_hotkeys():
    # keyboard.on_press(kb_callback)
    for key in hotkeys:
        keyboard.add_hotkey(key, kb_callback, args=(key,))


def remove_hotkeys():
    for key in hotkeys:
        keyboard.remove_hotkey(key)


if __name__ == '__main__':
    _exit = 0
    add_hotkeys()

    try:
        while not _exit:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    print('removing hotkeys')
    remove_hotkeys()