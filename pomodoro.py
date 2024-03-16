import paramparse
import time
from tqdm import tqdm
from win10toast import ToastNotifier
import winsound


class Params:
    sound = ""
    # sound = "Rhea.wav"
    duration = 5
    interval = 360
    n_intervals = 1000
    freq = 5000  # Hz
    threaded = 0


def main():
    params = Params()
    paramparse.process(params)

    if params.sound:
        winsound.PlaySound(params.sound, winsound.SND_FILENAME)

    toaster = ToastNotifier()
    toaster.show_toast(
        title=f"Pomodoro interval timer",
        msg=f"starting timer for {params.n_intervals} intervals of {params.interval} secs each",
        icon_path="pomodoro.ico",
        duration=3,
        threaded=params.threaded,
    )

    for i in range(params.n_intervals):

        for _ in tqdm(range(params.interval), desc=f'interval {i + 1} / {params.n_intervals}', ncols=100):
            time.sleep(1)

        toaster = ToastNotifier()
        if params.sound:
            # winsound.Beep(params.freq, 1000)
            # winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
            winsound.PlaySound(params.sound, winsound.SND_FILENAME)

        msg_ = f"interval {i + 1} / {params.n_intervals} completed"
        msg = '\n'.join([msg_, ]*4)
        toaster.show_toast(
            title=f"Pomodoro interval timer",
            msg=msg,
            icon_path="pomodoro.ico",
            duration=params.duration,
            threaded=params.threaded,
        )


if __name__ == '__main__':
    main()
