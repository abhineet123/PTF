import cv2
import sys
import time
import os, shutil
from datetime import datetime
from pprint import pformat
from tqdm import tqdm
import skvideo.io

import paramparse

from Misc import sortKey, resizeAR, sizeAR, move_or_del_files
from video_io import VideoWriterGPU


def read_image(src_path, filename):
    file_path = os.path.join(src_path, filename)
    assert os.path.exists(file_path), 'Image file {} does not exist'.format(file_path)

    image = cv2.imread(file_path)

    assert image is not None, 'Image could not be read: {}'.format(file_path)

    return image


def main():
    params = {
        'src_path': '.',
        'save_path': '',
        'save_root_dir': '',
        'img_ext': 'jpg',
        'show_img': 1,
        'del_src': 0,
        'start_id': 0,
        'n_frames': 0,
        'width': 0,
        'height': 0,
        'fps': 30,
        # 'codec': 'FFV1',
        # 'ext': 'avi',
        'codec': 'H264',
        'ext': 'mkv',
        'out_postfix': '',
        'reverse': 0,
        'move_src': 0,
        'use_skv': 0,
        'disable_suffix': 0,
        'read_in_batch': 1,
        'placement_type': 1,
        'recursive': 0,
    }

    paramparse.process_dict(params)
    _src_path = params['src_path']
    save_path = params['save_path']
    img_ext = params['img_ext']
    show_img = params['show_img']
    del_src = params['del_src']
    start_id = params['start_id']
    n_frames = params['n_frames']
    __width = params['width']
    __height = params['height']
    fps = params['fps']
    use_skv = params['use_skv']
    codec = params['codec']
    ext = params['ext']
    out_postfix = params['out_postfix']
    reverse = params['reverse']
    save_root_dir = params['save_root_dir']
    move_src = params['move_src']
    disable_suffix = params['disable_suffix']
    read_in_batch = params['read_in_batch']
    placement_type = params['placement_type']
    recursive = params['recursive']

    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

    src_root_dir = ''

    if os.path.isdir(_src_path):
        if recursive:
            src_paths = [_src_path]
        else:
            src_files = [k for k in os.listdir(_src_path) for _ext in img_exts if k.endswith(_ext)]
            if not src_files:
                # src_paths = [os.path.join(_src_path, k) for k in os.listdir(_src_path) if
                #              os.path.isdir(os.path.join(_src_path, k))]
                src_paths_gen = [[os.path.join(dirpath, d) for d in dirnames if
                                  any([os.path.splitext(f.lower())[1] in img_exts
                                       for f in os.listdir(os.path.join(dirpath, d))])]
                                 for (dirpath, dirnames, filenames) in os.walk(_src_path, followlinks=True)]
                src_paths = [item for sublist in src_paths_gen for item in sublist]
                src_root_dir = os.path.abspath(_src_path)
            else:
                src_paths = [_src_path]
        print('Found {} image sequence(s):\n{}'.format(len(src_paths), pformat(src_paths)))
    elif os.path.isfile(_src_path):
        print('Reading source image sequences from: {}'.format(_src_path))
        src_paths = [x.strip() for x in open(_src_path).readlines() if x.strip()]
        n_seq = len(src_paths)
        if n_seq <= 0:
            raise SystemError('No input sequences found in {}'.format(_src_path))
        print('n_seq: {}'.format(n_seq))
    else:
        raise IOError('Invalid src_path: {}'.format(_src_path))

    if recursive:
        print('searching for images recursively')

    if reverse == 1:
        print('Writing the reverse sequence')
    elif reverse == 2:
        print('Appending the reverse sequence')

    print('placement_type: {}'.format(placement_type))

    exit_prog = 0

    n_src_paths = len(src_paths)

    cwd = os.getcwd()

    for src_id, src_path in enumerate(src_paths):
        seq_name = os.path.basename(src_path)

        print('\n{}/{} Reading source images from: {}'.format(src_id + 1, n_src_paths, src_path))

        src_path = os.path.abspath(src_path)

        if move_src:
            rel_src_path = os.path.relpath(src_path, os.getcwd())
            dst_path = os.path.join(cwd, 'i2v', rel_src_path)
        else:
            dst_path = ''

        if recursive:
            src_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                             os.path.splitext(f.lower())[1] in img_exts]
                            for (dirpath, dirnames, filenames) in os.walk(src_path, followlinks=True)]
            src_files = [item for sublist in src_file_gen for item in sublist]
        else:
            src_files = [k for k in os.listdir(src_path) for _ext in img_exts if k.endswith(_ext)]
        n_src_files = len(src_files)
        if n_src_files <= 0:
            raise SystemError('No input frames found')
        src_files.sort(key=sortKey)
        print('n_src_files: {}'.format(n_src_files))

        if reverse == 1:
            src_files = src_files[::-1]
        elif reverse == 2:
            src_files += src_files[::-1]
            n_src_files *= 2

        _width, _height = __width, __height

        if os.path.exists(save_path):
            dst_mtime = os.path.getmtime(save_path)
            src_mtime = os.path.getmtime(src_path)

            dst_mtime_fmt = datetime.fromtimestamp(dst_mtime).strftime('%Y-%m-%d %H:%M:%S')
            src_mtime_fmt = datetime.fromtimestamp(src_mtime).strftime('%Y-%m-%d %H:%M:%S')

            print('Output video file already exists: {}'.format(save_path))

            if dst_mtime >= src_mtime:
                print('Last modified time: {} is not older than the source: {} so skipping it'.format(
                    dst_mtime_fmt, src_mtime_fmt
                ))
                save_path = ''
                continue
            else:
                print('Last modified time: {} is older than the source: {} so overwriting it'.format(
                    dst_mtime_fmt, src_mtime_fmt
                ))

        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        src_images = []
        print('orig: {} x {}'.format(_width, _height))

        if read_in_batch:
            print('reading all images in batch')
            src_images = [read_image(src_path, k) for k in tqdm(src_files)]
            if _height <= 0 and _width <= 0:
                heights, widths = zip(*[k.shape[:2] for k in src_images])
                _height, _width = max(heights), max(widths)
            elif _height <= 0:
                _height, _width = sizeAR(src_images[0], width=_width)
            elif _width <= 0:
                _height, _width = sizeAR(src_images[0], height=_height)

        else:
            if _height <= 0 or _width <= 0:
                temp_img = cv2.imread(os.path.join(src_path, src_files[0]))
                _height, _width, _ = temp_img.shape

        print('inferred: {} x {}'.format(_width, _height))

        if not save_path:
            save_fname = os.path.basename(src_path)

            if not disable_suffix:
                save_fname = '{}_{}'.format(save_fname, fps)

                if _height > 0 and _width > 0:
                    save_fname = '{}_{}x{}'.format(save_fname, _width, _height)

                if out_postfix:
                    save_fname = '{}_{}'.format(save_fname, out_postfix)

                if reverse:
                    save_fname = '{}_r{}'.format(save_fname, reverse)

            save_path = os.path.join(os.path.dirname(src_path), '{}.{}'.format(save_fname, ext))

            if src_root_dir and save_root_dir:
                save_path = save_path.replace(src_root_dir, save_root_dir)
                print('save_path: {}'.format(save_path))
                print('src_root_dir: {}'.format(src_root_dir))
                print('save_root_dir: {}'.format(save_root_dir))
                print('save_path: {}'.format(save_path))
                # sys.exit()

        if use_skv:
            video_out = skvideo.io.FFmpegWriter(save_path, outputdict={
                '-vcodec': 'libx264',  # use the h.264 codec
                '-crf': '0',  # set the constant rate factor to 0, which is lossless
                '-preset': 'veryslow'  # the slower the better compression, in princple, try
                # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
            })
        elif codec == 'H265':
            video_out = VideoWriterGPU(save_path, fps, (_width, _height))
        else:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_out = cv2.VideoWriter(save_path, fourcc, fps, (_width, _height))

        if video_out is None:
            raise IOError('Output video file could not be opened: {}'.format(save_path))

        print('Saving {}x{} output video to {}'.format(_width, _height, save_path))

        frame_id = start_id
        pause_after_frame = 0
        print_diff = max(1, int(n_src_files / 100))
        start_t = time.time()
        while True:
            if read_in_batch:
                image = src_images[frame_id]
            else:
                filename = src_files[frame_id]
                file_path = os.path.join(src_path, filename)
                if not os.path.exists(file_path):
                    raise SystemError('Image file {} does not exist'.format(file_path))

                image = cv2.imread(file_path)

            image = resizeAR(image, _width, _height, placement_type=placement_type)

            if show_img:
                cv2.imshow(seq_name, image)
                k = cv2.waitKey(1 - pause_after_frame) & 0xFF
                if k == 27:
                    exit_prog = 1
                    break
                elif k == ord('q'):
                    break
                elif k == 32:
                    pause_after_frame = 1 - pause_after_frame

            if use_skv:
                video_out.writeFrame(image[:, :, ::-1])  # write the frame as RGB not BGR
            else:
                video_out.write(image)

            frame_id += 1

            if frame_id % print_diff == 0:
                end_t = time.time()
                try:
                    proc_fps = float(print_diff) / (end_t - start_t)
                except:
                    proc_fps = 0

                sys.stdout.write('\rDone {:d}/{:d} frames at {:.4f} fps'.format(
                    frame_id - start_id, n_src_files - start_id, proc_fps))
                sys.stdout.flush()
                start_t = end_t

            if (frame_id - start_id) >= n_frames > 0:
                break

            if frame_id >= n_src_files:
                break

        sys.stdout.write('\n')
        sys.stdout.flush()

        if use_skv:
            video_out.close()  # close the writer
        else:
            video_out.release()

        if show_img:
            cv2.destroyWindow(seq_name)

        if move_src or del_src:
            move_or_del_files(src_path, src_files, dst_path)

        save_path = ''

        if exit_prog:
            break


if __name__ == '__main__':
    main()
