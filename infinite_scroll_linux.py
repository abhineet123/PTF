import time
import pyautogui
import os, sys

from Misc import processArguments

def main():
    params = {
        'max_t': 120,
        'relative': 0,
    }
    processArguments(sys.argv[1:], params)

    time.sleep(3)

    max_t = params['max_t']


    start_t = time.time()
    while True:
        try:
            pyautogui.press("pagedown")
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
