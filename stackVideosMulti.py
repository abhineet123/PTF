import cv2
import sys
import os, shutil
from pprint import pformat
from datetime import datetime
from Misc import resizeAR, stackImages, sizeAR
from ImageSequenceIO import ImageSequenceCapture, ImageSequenceWriter

import paramparse


#
# params = {
#     'root_dirs': [],
#     'annotations': [],
#     'root_dir': '',
#     'save_path': '',
#     'img_ext': 'jpg',
#     'show_img': 2,
#     'del_src': 0,
#     'start_id': 0,
#     'n_frames': 0,
#     'ann_size': 2,
#     # font_type,loc(x,y),size,thickness,col(r,g,b),bgr_col(r,g,b)
#     'ann_fmt': (0, 5, 45, 3, 2, 255, 255, 255, 0, 0, 0),
#     'preserve_order': 1,
#     'borderless': 0,
#     'only_height': 1,
#     'width': 0,
#     'height': 0,
#     'sep_size': 0,
#     'fps': 30,
#     'codec': 'H264',
#     'ext': 'jpg',
#     'grid_size': '',
#     'resize_factor': 1.0,
#     'recursive': 0,
# }
# processArguments(sys.argv[1:], params)


class Params:
    """
    :ivar ann_fmt:
    :type ann_fmt: tuple

    :ivar ann_size:
    :type ann_size: int

    :ivar annotations:
    :type annotations: list

    :ivar borderless:
    :type borderless: int

    :ivar codec:
    :type codec: str

    :ivar del_src:
    :type del_src: int

    :ivar ext:
    :type ext: str

    :ivar fps:
    :type fps: int

    :ivar grid_size:
    :type grid_size: str

    :ivar height:
    :type height: int

    :ivar img_ext:
    :type img_ext: str

    :ivar n_frames:
    :type n_frames: int

    :ivar only_height:
    :type only_height: int

    :ivar preserve_order:
    :type preserve_order: int

    :ivar recursive:
    :type recursive: int

    :ivar resize_factor:
    :type resize_factor: float

    :ivar root_dir:
    :type root_dir: str

    :ivar root_dirs:
    :type root_dirs: list

    :ivar save_path:
    :type save_path: str

    :ivar sep_size:
    :type sep_size: int

    :ivar show_img:
    :type show_img: int

    :ivar start_id:
    :type start_id: int

    :ivar width:
    :type width: int

    """

    def __init__(self):
        self.cfg = ()
        self.ann_fmt = (0, 5, 45, 3, 2, 255, 255, 255, 0, 0, 0)
        self.ann_size = 2
        self.annotations = []
        self.borderless = 0
        self.codec = 'H264'
        self.del_src = 0
        self.ext = 'jpg'
        self.out_width = 0
        self.out_height = 0
        self.fps = 30
        self.grid_size = ''
        self.height = 0
        self.img_ext = 'jpg'
        self.n_frames = 0
        self.only_height = 1
        self.preserve_order = 1
        self.recursive = 0
        self.img_seq = 1
        self.resize_factor = 1.0
        self.root_dirs = []
        self.save_path = ''
        self.sep_size = 0
        self.show_img = 1
        self.start_id = 0
        self.width = 0
        self.match_images = 1


def main():
    params = Params()

    paramparse.process(params)

    root_dirs = params.root_dirs
    annotations = params.annotations
    save_path = params.save_path
    img_ext = params.img_ext
    show_img = params.show_img
    del_src = params.del_src
    start_id = params.start_id
    n_frames = params.n_frames
    width = params.width
    height = params.height
    fps = params.fps
    codec = params.codec
    ext = params.ext
    out_width = params.out_width
    out_height = params.out_height
    grid_size = params.grid_size
    sep_size = params.sep_size
    only_height = params.only_height
    borderless = params.borderless
    preserve_order = params.preserve_order
    ann_fmt = params.ann_fmt
    resize_factor = params.resize_factor
    recursive = params.recursive
    img_seq = params.img_seq
    match_images = params.match_images

    if match_images:
        assert img_seq, "image matching is only supported in image sequence mode"

    vid_exts = ['.mkv', '.mp4', '.avi', '.mjpg', '.wmv']
    image_exts = ['.jpg', '.bmp', '.png', '.tif']

    min_n_sources = None
    min_sources = None
    min_sources_id = None

    sources_list = []
    for root_dir_id, root_dir in enumerate(root_dirs):
        root_dir = os.path.abspath(root_dir)
        if img_seq:
            sources = [k for k in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, k))]
        else:
            sources = [k for k in os.listdir(root_dir) if os.path.splitext(k)[1] in vid_exts]
        sources.sort()

        n_sources = len(sources)
        if min_n_sources is None or n_sources < min_n_sources:
            min_n_sources = n_sources
            min_sources = sources
            min_sources_id = root_dir_id

        sources = [os.path.join(root_dir, k) for k in sources]

        sources_list.append(sources)

    if match_images:
        for sources_id, sources in enumerate(sources_list):
            if sources_id == min_sources_id:
                continue

            sources = [k for k in sources if os.path.basename(k) in min_sources]

            assert len(sources) == min_n_sources, "invalid sources after filtering {}".format(sources)

            sources_list[sources_id] = sources

    src_paths = list(zip(*sources_list))

    print('sources_list:\n{}'.format(pformat(sources_list)))
    print('src_paths:\n{}'.format(pformat(src_paths)))

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    _exit = 0

    for _src_path in src_paths:
        _annotations = annotations
        _save_path = save_path
        _grid_size = grid_size
        n_frames = 0

        src_files = _src_path

        n_videos = len(src_files)
        assert n_videos > 0, 'no input videos found'

        if not _save_path:
            seq_dir = os.path.dirname(src_files[0])
            seq_name = os.path.splitext(os.path.basename(src_files[0]))[0]
            dst_path = os.path.join(seq_dir, 'stacked_{}'.format(timestamp), '{}.{}'.format(seq_name, ext))
        else:
            out_seq_name, out_ext = os.path.splitext(os.path.basename(_save_path))
            dst_path = os.path.join(os.path.dirname(_save_path), '{}_{}{}'.format(
                out_seq_name, datetime.now().strftime("%y%m%d_%H%M%S"), out_ext))

        save_dir = os.path.dirname(dst_path)
        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        print('Stacking: {} videos:'.format(n_videos))
        print('src_files:\n{}'.format(pformat(src_files)))

        if _annotations:
            if len(_annotations) == 1 and _annotations[0] == 1:
                _annotations = []
                for i in range(n_videos):
                    _annotations.append(seq_names[i])
            else:
                assert len(_annotations) == n_videos, 'Invalid annotations: {}'.format(_annotations)

                for i in range(n_videos):
                    if _annotations[i] == '__n__':
                        _annotations[i] = ''

            print('Adding annotations:\n{}'.format(pformat(_annotations)))
        else:
            _annotations = None

        if not _grid_size:
            _grid_size = None
        else:
            _grid_size = [int(x) for x in _grid_size.split('x')]
            if len(_grid_size) != 2 or _grid_size[0] * _grid_size[1] != n_videos:
                raise AssertionError('Invalid grid_size: {}'.format(_grid_size))

        n_frames_list = []
        cap_list = []
        size_list = []
        seq_names = []

        min_n_frames = None
        min_n_frames_id = 0

        for src_id, src_file in enumerate(src_files):
            src_file = os.path.abspath(src_file)
            seq_name = os.path.splitext(os.path.basename(src_file))[0]

            seq_names.append(seq_name)

            if os.path.isfile(src_file):
                cap = cv2.VideoCapture()
            elif os.path.isdir(src_file):
                cap = ImageSequenceCapture(src_file, recursive=recursive)
            else:
                raise IOError('Invalid src_file: {}'.format(src_file))

            if not cap.open(src_file):
                raise IOError('The video file ' + src_file + ' could not be opened')

            cv_prop = cv2.CAP_PROP_FRAME_COUNT
            h_prop = cv2.CAP_PROP_FRAME_HEIGHT
            w_prop = cv2.CAP_PROP_FRAME_WIDTH

            total_frames = int(cap.get(cv_prop))
            _height = int(cap.get(h_prop))
            _width = int(cap.get(w_prop))

            cap_list.append(cap)
            n_frames_list.append(total_frames)
            if min_n_frames is None or total_frames < min_n_frames:
                min_n_frames = total_frames
                min_n_frames_id = src_id

            size_list.append((_width, _height))

        if match_images:
            assert all(seq_name == seq_names[0] for seq_name in seq_names), "mismatch in seq_names: {}".format(seq_names)

            frames_list = [os.path.basename(k) for k in cap_list[min_n_frames_id].src_files]
            for src_id, cap in enumerate(cap_list):
                if src_id == min_n_frames_id:
                    continue
                cap_list[src_id].filter_files(frames_list)
                n_frames_list[src_id] = min_n_frames

        frame_id = start_id
        pause_after_frame = 0
        video_out = None

        win_name = 'stacked_{}'.format(datetime.now().strftime("%y%m%d_%H%M%S"))

        min_n_frames = min(n_frames_list)
        max_n_frames = max(n_frames_list)

        if n_frames <= 0:
            n_frames = max_n_frames
        else:
            if max_n_frames < n_frames:
                raise IOError(
                    'Invalid n_frames: {} for sequence list with max_n_frames: {}'.format(n_frames, max_n_frames))

        if show_img == 2:
            vis_only = True
            print('Running in visualization only mode')
        else:
            vis_only = False

        while True:

            images = []
            valid_caps = []
            valid_annotations = []
            for cap_id, cap in enumerate(cap_list):
                ret, image = cap.read()
                if not ret:
                    print('\nFrame {:d} could not be read'.format(frame_id + 1))
                    continue
                images.append(image)
                valid_caps.append(cap)
                if _annotations:
                    valid_annotations.append(_annotations[cap_id])

            cap_list = valid_caps
            if _annotations:
                _annotations = valid_annotations

            # if len(images) != n_videos:
            #     break

            frame_id += 1

            if frame_id <= start_id:
                break

            out_img = stackImages(images, _grid_size, borderless=borderless, preserve_order=preserve_order,
                                  annotations=_annotations, ann_fmt=ann_fmt, only_height=only_height, sep_size=sep_size)
            if resize_factor != 1:
                out_img = cv2.resize(out_img, (0, 0), fx=resize_factor, fy=resize_factor)

            if not vis_only:
                if video_out is None:
                    dst_height, dst_width = sizeAR(out_img, width=out_width, height=out_height)

                    if '.' + ext in vid_exts:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        video_out = cv2.VideoWriter(dst_path, fourcc, fps, (dst_width, dst_height))
                    elif '.' + ext in image_exts:
                        video_out = ImageSequenceWriter(dst_path, height=dst_height, width=dst_width)
                    else:
                        raise IOError('Invalid ext: {}'.format(ext))

                    if video_out is None:
                        raise IOError('Output video file could not be opened: {}'.format(dst_path))

                    print('Saving {}x{} output video to {}'.format(dst_width, dst_height, dst_path))

                out_img = resizeAR(out_img, width=dst_width, height=dst_height)
                video_out.write(out_img)

            if show_img:
                # out_img_disp = out_img
                out_img_disp = resizeAR(out_img, 1280)
                cv2.imshow(win_name, out_img_disp)
                k = cv2.waitKey(1 - pause_after_frame) & 0xFF
                if k == ord('q'):
                    _exit = 1
                    break
                elif k == 27:
                    break
                elif k == 32:
                    pause_after_frame = 1 - pause_after_frame

            sys.stdout.write('\rDone {:d}/{:d} frames '.format(frame_id - start_id, n_frames))
            sys.stdout.flush()

            if frame_id - start_id >= n_frames:
                break

        if _exit:
            break

        sys.stdout.write('\n')
        sys.stdout.flush()

        video_out.release()

        if show_img:
            cv2.destroyWindow(win_name)


if __name__ == '__main__':
    main()
