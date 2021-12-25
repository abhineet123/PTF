import cv2
import sys
import os, shutil
from pprint import pprint
from datetime import datetime
from Misc import processArguments, sortKey, resizeAR, stackImages
from ImageSequenceIO import ImageSequenceCapture, ImageSequenceWriter

params = {
    'src_paths': [],
    'annotations': [],
    'root_dir': '',
    'save_path': '',
    'img_ext': 'jpg',
    'show_img': 1,
    'del_src': 0,
    'start_id': 0,
    'n_frames': 0,
    'ann_size': 2,
    # font_type,loc(x,y),size,thickness,col(r,g,b),bgr_col(r,g,b)
    'ann_fmt': (0, 5, 45, 3, 2, 255, 255, 255, 0, 0, 0),
    'preserve_order': 1,
    'borderless': 0,
    'only_height': 1,
    'width': 0,
    'height': 0,
    'sep_size': 0,
    'fps': 30,
    'codec': 'H264',
    'ext': 'jpg',
    'grid_size': '',
    'resize_factor': 1.0,
}

processArguments(sys.argv[1:], params)
_src_path = params['src_paths']
annotations = params['annotations']
root_dir = params['root_dir']
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
grid_size = params['grid_size']
sep_size = params['sep_size']
only_height = params['only_height']
borderless = params['borderless']
preserve_order = params['preserve_order']
ann_fmt = params['ann_fmt']
resize_factor = params['resize_factor']

vid_exts = ['mkv', 'mp4', 'avi', 'mjpg', 'wmv']
image_exts = ['jpg', 'bmp', 'png', 'tif']

if len(_src_path) == 1:
    _src_path = _src_path[0]
    print('Reading source videos from: {}'.format(_src_path))
    if os.path.isdir(_src_path):
        src_files = [os.path.join(_src_path, k) for k in os.listdir(_src_path) for _ext in vid_exts if
                     k.endswith('.{}'.format(_ext))]
        src_files.sort(key=sortKey)
    else:
        src_files = [x.strip() for x in open(_src_path).readlines() if x.strip()]
else:
    src_files = _src_path

if root_dir:
    src_files = [os.path.join(root_dir, name) for name in src_files]

if not save_path:
    dst_path = os.path.join(os.path.dirname(src_files[0]), 'stacked',
                            '{}.{}'.format(datetime.now().strftime("%y%m%d_%H%M%S"), ext))
else:
    out_seq_name, out_ext = os.path.splitext(os.path.basename(save_path))
    dst_path = os.path.join(os.path.dirname(save_path), '{}_{}{}'.format(
        out_seq_name, datetime.now().strftime("%y%m%d_%H%M%S"), out_ext))

save_dir = os.path.dirname(dst_path)
if save_dir and not os.path.isdir(save_dir):
    os.makedirs(save_dir)

n_videos = len(src_files)
if n_videos <= 0:
    raise SystemError('No input videos found')
print('Stacking: {} videos:'.format(n_videos))
pprint(src_files)

if not grid_size:
    grid_size = None
else:
    grid_size = [int(x) for x in grid_size.split('x')]
    if len(grid_size) != 2 or grid_size[0] * grid_size[1] != n_videos:
        raise IOError('Invalid grid_size: {}'.format(grid_size))

n_frames_list = []
cap_list = []
size_list = []

for src_file in src_files:
    src_file = os.path.abspath(src_file)
    seq_name = os.path.splitext(os.path.basename(src_file))[0]
    if os.path.isfile(src_file):
        cap = cv2.VideoCapture()
    elif os.path.isdir(src_file):
        cap = ImageSequenceCapture(src_file)
    else:
        raise IOError('Invalid src_file: {}'.format(src_file))

    if not cap.open(src_file):
        raise IOError('The video file ' + src_file + ' could not be opened')

    if cv2.__version__.startswith('2'):
        cv_prop = cv2.cv.CAP_PROP_FRAME_COUNT
        h_prop = cv2.cv.CAP_PROP_FRAME_HEIGHT
        w_prop = cv2.cv.CAP_PROP_FRAME_WIDTH
    else:
        cv_prop = cv2.CAP_PROP_FRAME_COUNT
        h_prop = cv2.CAP_PROP_FRAME_HEIGHT
        w_prop = cv2.CAP_PROP_FRAME_WIDTH

    total_frames = int(cap.get(cv_prop))
    _height = int(cap.get(h_prop))
    _width = int(cap.get(w_prop))

    cap_list.append(cap)
    n_frames_list.append(total_frames)
    size_list.append((_width, _height))

frame_id = start_id
pause_after_frame = 0
video_out = None

win_name = 'stacked_{}'.format(datetime.now().strftime("%y%m%d_%H%M%S"))

min_n_frames = min(n_frames_list)

if n_frames <= 0:
    n_frames = min_n_frames
else:
    if min_n_frames < n_frames:
        raise IOError('Invalid n_frames: {} for sequence list with min_n_frames: {}'.format(n_frames, min_n_frames))

if annotations:
    if len(annotations) != n_videos:
        raise IOError('Invalid annotations: {}'.format(annotations))
    for i in range(n_videos):
        if annotations[i] == '__n__':
            annotations[i] = ''
    print('Adding annotations:')
    pprint(annotations)
else:
    annotations = None

while True:

    images = []
    for cap in cap_list:
        ret, image = cap.read()
        if not ret:
            print('\nFrame {:d} could not be read'.format(frame_id + 1))
            break
        images.append(image)

    if len(images) != n_videos:
        break

    frame_id += 1

    if frame_id <= start_id:
        break

    out_img = stackImages(images, grid_size, borderless=borderless, preserve_order=preserve_order,
                          annotations=annotations, ann_fmt=ann_fmt, only_height=only_height, sep_size=sep_size)
    if resize_factor != 1:
        out_img = cv2.resize(out_img, (0, 0), fx=resize_factor, fy=resize_factor)

    if video_out is None:
        dst_height, dst_width = out_img.shape[:2]

        if ext in vid_exts:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_out = cv2.VideoWriter(dst_path, fourcc, fps, (dst_width, dst_height))
        elif ext in image_exts:
            video_out = ImageSequenceWriter(dst_path)
        else:
            raise IOError('Invalid ext: {}'.format(ext))

        if video_out is None:
            raise IOError('Output video file could not be opened: {}'.format(dst_path))

        print('Saving {}x{} output video to {}'.format(dst_width, dst_height, dst_path))

    video_out.write(out_img)
    if show_img:
        out_img_disp = resizeAR(out_img, 1920, 1080)
        cv2.imshow(win_name, out_img_disp)
        k = cv2.waitKey(1 - pause_after_frame) & 0xFF
        if k == ord('q') or k == 27:
            break
        elif k == 32:
            pause_after_frame = 1 - pause_after_frame

    sys.stdout.write('\rDone {:d}/{:d} frames '.format(frame_id - start_id, n_frames))
    sys.stdout.flush()

    if frame_id - start_id >= n_frames:
        break

sys.stdout.write('\n')
sys.stdout.flush()

video_out.release()

if show_img:
    cv2.destroyWindow(win_name)
