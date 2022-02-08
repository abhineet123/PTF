import os
import cv2

from paramparse import process

from Misc import resizeAR


class Params:
    def __init__(self):
        self.cfg = ('',)
        self.src_path = ''
        self.out_size = 300
        self.out_w = 0
        self.out_h = 0


def main():
    params = Params()
    process(params)

    src_path = params.src_path

    if not src_path:
        try:
            from Tkinter import Tk
        except ImportError:
            from tkinter import Tk
        try:
            src_path = Tk().clipboard_get()
        except BaseException as e:
            print('Tk().clipboard_get() failed: {}'.format(e))
            return

    src_path = src_path.replace(os.sep, '/').replace('"', '')
    assert os.path.exists(src_path), "src_path does not exist: {}".format(src_path)

    src_dir = os.path.dirname(src_path)
    src_name = os.path.basename(src_path)

    src_name_noext, src_ext = os.path.splitext(src_name)

    out_name = src_name_noext + '_' + src_ext

    out_path = os.path.join(src_dir, out_name)

    assert not os.path.exists(out_path), "out_path already exists"

    img = cv2.imread(src_path)

    img_h, img_w = img.shape[:2]

    out_h = out_w = 0

    if img_h > img_w:
        out_h = params.out_size
    else:
        out_w = params.out_size

    resized_img = resizeAR(img, width=out_w, height=out_h)

    cv2.imwrite(out_path, resized_img)


if __name__ == '__main__':
    main()
