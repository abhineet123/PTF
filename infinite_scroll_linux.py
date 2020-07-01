import time
# import pyautogui
import paramparse
import os, sys
from subprocess import Popen, PIPE


def keypress(sequence):
    p = Popen(['xte'], stdin=PIPE)
    p.communicate(input=sequence)

def main():

    params = {
        'max_t': 120,
        'relative': 0,
    }
    paramparse.process_dict(params)
    max_t = params['max_t']

    time.sleep(3)

    start_t = time.time()
    while True:
        try:
            keypress(b'Page_Down')
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
