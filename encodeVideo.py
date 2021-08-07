import cv2
import sys
import os, shutil

import paramparse

from Misc import sortKey, resizeAR

params = {
    'src_path': '.',
    'save_path': '',
    'img_ext': 'jpg',
    'show_img': 1,
    'del_src': 0,
    'start_id': 0,
    'n_frames': 0,
    'reverse': 0,
    'combine': 0,
    'res': '',
    'fps': 30,
    'codec': 'H264',
    'ext': 'mkv',
    'out_postfix': '',
    'add_headers': 0.0,
    'vid_exts': ['.mkv', '.mp4', '.avi', '.mjpg', '.wmv'],
    'img_exts': ['.jpg', '.png', '.jpeg', '.tif', '.bmp'],
}

paramparse.process_dict(params)

_src_path = params['src_path']
save_path = params['save_path']
img_ext = params['img_ext']
show_img = params['show_img']
del_src = params['del_src']
start_id = params['start_id']
n_frames = params['n_frames']
res = params['res']
fps = params['fps']
codec = params['codec']
ext = params['ext']
reverse = params['reverse']
combine = params['combine']
out_postfix = params['out_postfix']
add_headers = params['add_headers']
vid_exts = params['vid_exts']
img_exts = params['img_exts']

height = width = 0
if res:
    width, height = [int(x) for x in res.split('x')]

print('Reading source videos from: {}'.format(_src_path))

header_files = None

if os.path.isdir(_src_path):
    src_files = [os.path.join(_src_path, k) for k in os.listdir(_src_path) for _ext in vid_exts if k.endswith(_ext)]
    src_files.sort(key=sortKey)
    header_root = _src_path
else:
    src_files = [_src_path]
    header_root = os.path.dirname(_src_path)

n_videos = len(src_files)
if n_videos <= 0:
    raise IOError('No input videos found')

print('n_videos: {}'.format(n_videos))

if add_headers > 0:
    header_files = [os.path.join(header_root, k) for k in os.listdir(header_root) for _ext in img_exts if
                    k.endswith(_ext)]
    header_files.sort(key=sortKey)
    n_headers = len(header_files)
    if n_headers != n_videos:
        print('Mismatch between n_headers: {} and n_videos: {}'.format(n_headers, n_videos))
        add_headers = 0

if add_headers > 0:
    print('Adding headers of length {} secs'.format(add_headers))

if reverse == 1:
    print('Writing reversed video')
elif reverse == 2:
    print('Appending reversed video')

if combine and n_videos > 1:
    print('Combining all videos into a single output video')

video_out = None

n_header_frames = int(add_headers * fps)

for src_id, _src_path in enumerate(src_files):

    src_path = os.path.abspath(_src_path)
    seq_name = os.path.splitext(os.path.basename(src_path))[0]

    if not save_path:
        dst_seq_name = seq_name
        if start_id > 0:
            dst_seq_name = '{}_{}'.format(dst_seq_name, start_id)
        if n_frames > 0:
            dst_seq_name = '{}_{}'.format(dst_seq_name, start_id + n_frames)
        dst_seq_name = '{}_{}'.format(dst_seq_name, fps)
        if reverse:
            dst_seq_name = '{}_r{}'.format(dst_seq_name, reverse)
        if out_postfix:
            dst_seq_name = '{}_{}'.format(dst_seq_name, out_postfix)
        if combine:
            dst_seq_name = '{}_combined'.format(dst_seq_name)
        dst_path = os.path.join(os.path.dirname(src_path), dst_seq_name + '.' + ext)
    else:
        dst_path = save_path

    if dst_path == src_path:
        print('Skipping {:s} as having the save name as its target'.format(src_path))

    save_dir = os.path.dirname(dst_path)
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture()
    if not cap.open(src_path):
        raise IOError('The video file ' + src_path + ' could not be opened')

    if cv2.__version__.startswith('3'):
        cv_prop = cv2.CAP_PROP_FRAME_COUNT
        h_prop = cv2.CAP_PROP_FRAME_HEIGHT
        w_prop = cv2.CAP_PROP_FRAME_WIDTH
    else:
        cv_prop = cv2.cv.CAP_PROP_FRAME_COUNT
        h_prop = cv2.cv.CAP_PROP_FRAME_HEIGHT
        w_prop = cv2.cv.CAP_PROP_FRAME_WIDTH

    total_frames = int(cap.get(cv_prop))
    _height = int(cap.get(h_prop))
    _width = int(cap.get(w_prop))

    if n_frames <= 0:
        dst_n_frames = total_frames
    else:
        if total_frames > 0 and n_frames > total_frames:
            raise AssertionError('Invalid n_frames {} for video with {} frames'.format(n_frames, total_frames))
        dst_n_frames = n_frames

    if height <= 0 or width <= 0:
        dst_height, dst_width = _height, _width
        enable_resize = 0
    else:
        enable_resize = 1
        dst_height, dst_width = height, width

    if video_out is None or not combine:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_out = cv2.VideoWriter(dst_path, fourcc, fps, (dst_width, dst_height))

        if video_out is None:
            raise IOError('Output video file could not be opened: {}'.format(dst_path))

        print('Saving {}x{} output video to {}'.format(dst_width, dst_height, dst_path))

    if start_id > 0:
        print('Starting from frame_id {}'.format(start_id))

    frame_id = 0
    pause_after_frame = 0
    frames = []

    if add_headers > 0:
        header_path = header_files[src_id]
        header_img = cv2.imread(header_path)
        header_img = resizeAR(header_img, dst_width, dst_height)
        for i in range(n_header_frames):
            video_out.write(header_img)

    while True:

        ret, image = cap.read()
        if not ret:
            print('\nFrame {:d} could not be read'.format(frame_id + 1))
            break

        frame_id += 1

        if frame_id <= start_id:
            continue

        if enable_resize:
            image = resizeAR(image, dst_width, dst_height)

        if show_img:
            cv2.imshow(seq_name, image)
            k = cv2.waitKey(1 - pause_after_frame) & 0xFF
            if k == ord('q') or k == 27:
                break
            elif k == 32:
                pause_after_frame = 1 - pause_after_frame

        if reverse:
            frames.append(image)

        if reverse != 1:
            video_out.write(image)

        sys.stdout.write('\rDone {:d} frames '.format(frame_id - start_id))
        sys.stdout.flush()

        if dst_n_frames > 0 and (frame_id - start_id) >= dst_n_frames:
            break

        if frame_id >= total_frames:
            break

    sys.stdout.write('\n')
    sys.stdout.flush()

    if reverse:
        for frame in frames[::-1]:
            if show_img:
                cv2.imshow(seq_name, frame)
                k = cv2.waitKey(1 - pause_after_frame) & 0xFF
                if k == ord('q') or k == 27:
                    break
                elif k == 32:
                    pause_after_frame = 1 - pause_after_frame
            video_out.write(frame)

    if not combine:
        video_out.release()

    if show_img:
        cv2.destroyWindow(seq_name)

    if del_src and src_path != dst_path:
        print('Removing source video {}'.format(src_path))
        os.remove(src_path)

if combine:
    video_out.release()
