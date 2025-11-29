import os
import shutil
import sys
from PIL import Image
import paramparse
from pdf2image import convert_from_path

class Params:
    def __init__(self):
        self.cfg = ()
        self.src_path = ''
        self.dpi = 100


def main():
    params = paramparse.process(Params)
    pdf_exts = ['pdf']

    if os.path.isdir(params.src_path):
        pdf_paths = [
            os.path.join(params.src_path, k) for k in os.listdir(params.src_path) for _ext in pdf_exts if
            k.lower().endswith('.{}'.format(_ext))]
    elif os.path.isfile(params.src_path):
        pdf_paths = [params.src_path,]
    else:
        raise IOError(f'nonexistent src_path: {params.src_path}')

    for pdf_path in pdf_paths:
        print('extracting images from pdf {}'.format(pdf_path))
        pdf_name = os.path.splitext(pdf_paths[0])[0]

        pages = convert_from_path(pdf_path, params.dpi)

        for page_id, page in enumerate(pages):
            out_path = f'{pdf_name}_{page_id}.jpg'
            page.save(out_path, format='JPEG', subsampling=0, quality=100)
            print()

if __name__ == '__main__':
    main()
