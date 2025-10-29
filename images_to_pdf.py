import os
import shutil
import sys
from PIL import Image
import paramparse


class Params:
    def __init__(self):
        self.cfg = ()
        self.src_path = '.'

def main():
    params = paramparse.process(Params)
    image_exts = ['jpg', 'bmp', 'png', 'tif']

    image_paths = [
        os.path.join(params.src_path, k) for k in os.listdir(params.src_path) for _ext in image_exts if
        k.endswith('.{}'.format(_ext))]
    n_images = len(image_paths)

    print('combining images into pdf {}'.format(image_paths))

    images = [Image.open(f) for f in image_paths]

    pdf_path = os.path.splitext(image_paths[0])[0] + '.pdf'

    images[0].save(
        pdf_path, "PDF", resolution=100.0, save_all=True, append_images=images[1:]
    )
        
if __name__ == '__main__':
    main()
