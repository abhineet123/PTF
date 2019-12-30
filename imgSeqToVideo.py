import cv2
import sys
import time
import os, shutil
from datetime import datetime
from pprint import pformat

from Misc import processArguments, sortKey, resizeAR
from video_io import VideoWriterGPU

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
}

processArguments(sys.argv[1:], params)
_src_path = params['src_path']
save_path = params['save_path']
img_ext = params['img_ext']
show_img = params['show_img']
del_src = params['del_src']
start_id = params['start_id']
n_frames = params['n_frames']
_width = params['width']
_height = params['height']
fps = params['fps']
codec = params['codec']
ext = params['ext']
out_postfix = params['out_postfix']
reverse = params['reverse']
save_root_dir = params['save_root_dir']

img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

src_root_dir = ''

if os.path.isdir(_src_path):
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

if reverse == 1:
    print('Writing the reverse sequence')
elif reverse == 2:
    print('Appending the reverse sequence')

exit_prog = 0

n_src_paths = len(src_paths)

for src_id, src_path in enumerate(src_paths):
    seq_name = os.path.basename(src_path)

    print('{}/{} Reading source images from: {}'.format(src_id + 1, n_src_paths, src_path))

    src_path = os.path.abspath(src_path)
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

    width, height = _width, _height

    if not save_path:
        save_fname = '{}_{}'.format(os.path.basename(src_path), fps)

        if height > 0 and width > 0:
            save_fname = '{}_{}x{}'.format(save_fname, width, height)

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

    if height <= 0 or width <= 0:
        temp_img = cv2.imread(os.path.join(src_path, src_files[0]))
        height, width, _ = temp_img.shape

    if codec == 'H265':
        video_out = VideoWriterGPU(save_path, fps, (width, height))
    else:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    if video_out is None:
        raise IOError('Output video file could not be opened: {}'.format(save_path))

    print('Saving {}x{} output video to {}'.format(width, height, save_path))

    frame_id = start_id
    pause_after_frame = 0
    print_diff = max(1, int(n_src_files / 100))
    start_t = time.time()
    while True:
        filename = src_files[frame_id]
        file_path = os.path.join(src_path, filename)
        if not os.path.exists(file_path):
            raise SystemError('Image file {} does not exist'.format(file_path))

        image = cv2.imread(file_path)

        image = resizeAR(image, width, height)

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

        video_out.write(image)

        frame_id += 1

        if frame_id % print_diff == 0:
            end_t = time.time()
            fps = float(print_diff) / (end_t - start_t)
            sys.stdout.write('\rDone {:d}/{:d} frames at {:.4f} fps'.format(
                frame_id - start_id, n_src_files - start_id, fps))
            sys.stdout.flush()
            start_t = end_t

        if n_frames > 0 and (frame_id - start_id) >= n_frames:
            break

        if frame_id >= n_src_files:
            break

    sys.stdout.write('\n\n')
    sys.stdout.flush()

    video_out.release()

    if show_img:
        cv2.destroyWindow(seq_name)

    if del_src:
        print('Removing source folder {}'.format(src_path))
        shutil.rmtree(src_path)

    save_path = ''

    if exit_prog:
        break
