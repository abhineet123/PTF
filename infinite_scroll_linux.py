import time
# import pyautogui
import paramparse
import os, sys
from subprocess import Popen, PIPE

"""apt-get install xautomation"""


def keypress(sequence):
    p = Popen(['xte'], stdin=PIPE)
    p.communicate(input=sequence)

def main():

    params = {
        'mode': 0,
        'switch_t': 3,
        'max_t': 120,
    }
    paramparse.process_dict(params)
    max_t = params['max_t']
    switch_t = params['switch_t']
    mode = params['mode']

    print(f'Waiting for {switch_t} seconds to allow switching to the target window')
    time.sleep(switch_t)

    if mode == 0:
        key = b'key Page_Down'
    else:
        key = b'key Return'


    start_t = time.time()
    while True:
        try:
            keypress(key)
            # pyautogui.press("pagedown")
        except BaseException as e:
            print('BaseException: {}'.format(e))
            break
        end_t = time.time()
        time_elapsed = end_t - start_t
        if time_elapsed > max_t:
            break
        print(time_elapsed)


if __name__ == '__main__':
    main()
