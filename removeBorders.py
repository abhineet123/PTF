import os
import cv2
import sys
import numpy as np
from Misc import processArguments

params = {
    'src_path': '.',
    'thresh': 0,
    'dst_path': '',
    'show_img': 0,
    'quality': 3,
    'border_type': 2,
    'resize': 0,
    'out_size': '',
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)
    src_path = params['src_path']
    thresh = params['thresh']
    dst_path = params['dst_path']
    show_img = params['show_img']
    quality = params['quality']
    border_type = params['border_type']
    # 0: LHS, 1:RHS, 2: both
    resize = params['resize']
    out_size = params['out_size']

    if out_size:
        resize = 1

    src_path = os.path.abspath(src_path)

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

    if border_type == 0:
        print('Removing only LHS borders')
    elif border_type == 1:
        print('Removing only RHS borders')
    else:
        print('Removing both LHS and RHS borders')

    img_id = 0

    if resize:
        if out_size:
            out_width, out_height = [int(x) for x in out_size.split('x')]
        else:
            src_img = cv2.imread(os.path.join(src_path, src_file_list[0]))
            out_height, out_width = src_img.shape[:2]
        print('Resizing output images to : {}x{}'.format(out_width, out_height))

    if not dst_path:
        dst_path = '{:s}_no_borders'.format(src_path)
        if resize:
            dst_path = '{:s}_{}x{}'.format(dst_path, out_width, out_height)

    print('Writing output images to: {}'.format(dst_path))
    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)

    for img_fname in src_file_list:

        src_img_fname = os.path.join(src_path, img_fname)
        src_img = cv2.imread(src_img_fname)
        img_fname_no_ext = os.path.splitext(img_fname)[0]

        if src_img is None:
            raise SystemError('Source image could not be read from: {}'.format(src_img_fname))

        src_height, src_width, n_channels = src_img.shape

        if border_type == 1:
            start_col_id = 0
        else:
            patch_size = 0
            while True:
                patch = src_img[:patch_size, :patch_size, :]
                if not np.all(patch <= thresh):
                    break
                patch_size += 1
            start_col_id = patch_size

        if border_type == 0:
            end_col_id = src_width
        else:
            patch_size = 0
            while True:
                patch = src_img[:patch_size, src_width - patch_size - 1:, :]
                if not np.all(patch <= thresh):
                    break
                patch_size += 1

            end_col_id = src_width - patch_size - 1

        dst_img = src_img[:, start_col_id:end_col_id, :].astype(np.uint8)
        if resize:
            dst_img = cv2.resize(dst_img, (out_width, out_height))
        dst_img_fname = os.path.join(dst_path, '{}.png'.format(img_fname_no_ext))
        cv2.imwrite(dst_img_fname, dst_img, [int(cv2.IMWRITE_PNG_COMPRESSION), quality])

        img_id += 1
        sys.stdout.write('\rDone {:d}/{:d} images'.format(img_id, total_frames))
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()
