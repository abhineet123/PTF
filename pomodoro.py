import os

import paramparse
import time
from tqdm import tqdm
from win10toast import ToastNotifier


class Params:

    def __init__(self):
        self.cfg = ()
        self.interval = 360
        self.n_intervals = 1000


def main():
    params = Params()
    paramparse.process(params)

    for i in range(params.n_intervals):

        for _ in tqdm(range(params.interval), desc=f'interval {i + 1}'):
            time.sleep(1)

        toaster = ToastNotifier()
        toaster.show_toast(f"interval {i + 1} / {params.n_intervals} completed",
                           icon_path="pomodoro.ico",
                           duration=2)


if __name__ == '__main__':
    main()
