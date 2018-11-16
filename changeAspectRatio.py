import os
import cv2
import sys
import numpy as np
from Misc import processArguments

params = {
    'src_path': '/home/abhineet/N/Datasets/617',
    'width': 16,
    'height': 9,
    'dst_path': '',
    'show_img': 0,
    'quality': 3,
    'resize': 0,
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)
    src_path = params['src_path']
    width = params['width']
    height = params['height']
    dst_path = params['dst_path']
    show_img = params['show_img']
    quality = params['quality']
    resize = params['resize']

    print('Reading source images from: {}'.format(src_path))

    img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff', '.gif')
    src_file_list = [k for k in os.listdir(src_path) if os.path.splitext(k.lower())[1] in img_exts]

    total_frames = len(src_file_list)
    if total_frames <= 0:
        raise SystemError('No input frames found')
    print('total_frames: {}'.format(total_frames))
    src_file_list.sort()

    # total_frames = len(src_file_list)
    # print('total_frames after sorting: {}'.format(total_frames))
    # sys.exit()

    aspect_ratio = float(width) / float(height)

    if not dst_path:
        dst_path = '{:s}_{:d}_{:d}'.format(src_path, width, height)

    print('Writing output images to: {}'.format(dst_path))
    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)

    img_id = 0

    if resize:
        print('Resizing images to {}x{}'.format(width, height))

    for img_fname in src_file_list:

        src_img_fname = os.path.join(src_path, img_fname)
        src_img = cv2.imread(src_img_fname)
        img_fname_no_ext = os.path.splitext(img_fname)[0]

        if src_img is None:
            raise SystemError('Source image could not be read from: {}'.format(src_img_fname))

        src_height, src_width, n_channels = src_img.shape
        src_aspect_ratio = float(src_width) / float(src_height)

        if src_aspect_ratio == aspect_ratio:
            dst_width = src_width
            dst_height = src_height
            start_row = start_col = 0
        elif src_aspect_ratio > aspect_ratio:
            dst_width = src_width
            dst_height = int(src_width / aspect_ratio)
            start_row = int((dst_height - src_height) / 2.0)
            start_col = 0
        else:
            dst_height = src_height
            dst_width = int(src_height * aspect_ratio)
            start_col = int((dst_width - src_width) / 2.0)
            start_row = 0

        dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=np.uint8)

        dst_img[start_row:start_row + src_height, start_col:start_col + src_width, :] = src_img
        dst_img_fname = os.path.join(dst_path, '{}.png'.format(img_fname_no_ext))

        if resize:
            dst_img = cv2.resize(dst_img, (width, height))

        cv2.imwrite(dst_img_fname, dst_img, [int(cv2.IMWRITE_PNG_COMPRESSION), quality])

        img_id += 1
        sys.stdout.write('\rDone {:d}/{:d} images'.format(img_id, total_frames))
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()
