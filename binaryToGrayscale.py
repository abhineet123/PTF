import cv2
import sys
import os, shutil
import numpy as np

from Misc import processArguments, sortKey, resizeAR

params = {
    'src_path': '.',
    'save_path': '',
    'img_ext': 'jpg',
    'show_img': 1,
    'del_src': 0,
    'start_id': 0,
    'n_frames': 0,
    'width': 0,
    'height': 0,
    'fps': 30,
    'codec': 'H264',
    'ext': 'mkv',
    'exp_base': 2.0,
    'resize_factor': 1.0,
}

processArguments(sys.argv[1:], params)
_src_path = params['src_path']
save_path = params['save_path']
img_ext = params['img_ext']
show_img = params['show_img']
del_src = params['del_src']
start_id = params['start_id']
n_frames = params['n_frames']
width = params['width']
height = params['height']
fps = params['fps']
codec = params['codec']
ext = params['ext']
exp_base = params['exp_base']
resize_factor = params['resize_factor']

img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

if os.path.isdir(_src_path):
    src_file_list = [k for k in os.listdir(_src_path) for _ext in img_exts if k.endswith(_ext)]
    if src_file_list:
        src_paths = [_src_path]
    else:
        src_paths = [os.path.join(_src_path, k) for k in os.listdir(_src_path) if
                     os.path.isdir(os.path.join(_src_path, k))]
elif os.path.isfile(_src_path):
    print('Reading source image sequences from: {}'.format(_src_path))
    src_paths = [x.strip() for x in open(_src_path).readlines() if x.strip()]
    n_seq = len(src_paths)
    if n_seq <= 0:
        raise SystemError('No input sequences found')
    print('n_seq: {}'.format(n_seq))
else:
    raise IOError('Invalid src_path: {}'.format(_src_path))

out_norm_factor = 255.0 * (exp_base - 1.0) / (exp_base ** 8 - 1)

for src_path in src_paths:
    seq_name = os.path.basename(src_path)
    src_win_name = '{}_src'.format(seq_name)
    dst_win_name = '{}_dst'.format(seq_name)

    print('Reading source images from: {}'.format(src_path))
    src_path = os.path.abspath(src_path)
    src_file_list = [k for k in os.listdir(src_path) for _ext in img_exts if k.endswith(_ext)]
    total_frames = len(src_file_list)
    if total_frames <= 0:
        raise SystemError('No input frames found')
    print('total_frames: {}'.format(total_frames))
    src_file_list.sort(key=sortKey)

    if not save_path:
        save_path = os.path.join(os.path.dirname(src_path), os.path.basename(src_path) + '_gs')

    if save_path and not os.path.isdir(save_path):
        os.makedirs(save_path)

    print('Writing output images to {}'.format(save_path))

    frame_id = start_id
    pause_after_frame = 0
    while True:
        filename = src_file_list[frame_id]
        file_path = os.path.join(src_path, filename)
        if not os.path.exists(file_path):
            raise SystemError('Image file {} does not exist'.format(file_path))

        src_img = cv2.imread(file_path)
        h, w = [int(d*resize_factor) for d in src_img.shape[:2]]

        w_rmd = w % 8
        # if w_rmd != 0:
        #     print('Resized image width {} is not a multiple of 8'.format(w))

        src_img = resizeAR(src_img, w - w_rmd, h)
        w -= w_rmd
        src_img = src_img[:, :, 0].squeeze()

        dst_h, dst_w = h, int(w / 8)

        # print('src_size: {}x{}'.format(w, h))
        # print('dst_size: {}x{}'.format(dst_w, dst_h))

        dst_img = np.zeros((dst_h, dst_w), dtype=np.float64)
        for r in range(dst_h):
            curr_col = 0
            for c in range(dst_w):
                curr_pix_val = 0
                for k in range(8):
                    src_pix = src_img[r, curr_col]
                    curr_col += 1

                    if src_pix > 127:
                        curr_pix_val += np.power(exp_base, k)

                # print('curr_pix_val: ', curr_pix_val)
                dst_img[r, c] = curr_pix_val
        dst_img = (dst_img * out_norm_factor).astype(np.uint8)
        
        dst_path = os.path.join(save_path, filename)
        cv2.imwrite(dst_path, dst_img)

        if show_img:
            cv2.imshow(src_win_name, src_img)
            cv2.imshow(dst_win_name, dst_img)
            k = cv2.waitKey(1 - pause_after_frame) & 0xFF
            if k == ord('q') or k == 27:
                break
            elif k == 32:
                pause_after_frame = 1 - pause_after_frame

        frame_id += 1
        sys.stdout.write('\rDone {:d} frames '.format(frame_id - start_id))
        sys.stdout.flush()

        if n_frames > 0 and (frame_id - start_id) >= n_frames:
            break

        if frame_id >= total_frames:
            break

    sys.stdout.write('\n\n')
    sys.stdout.flush()

    if show_img:
        cv2.destroyWindow(src_win_name)
        cv2.destroyWindow(dst_win_name)

    if del_src:
        print('Removing source folder {}'.format(src_path))
        shutil.rmtree(src_path)

    save_path = ''
