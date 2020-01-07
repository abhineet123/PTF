import os
import cv2
import sys
import time
import imageio
import numpy as np
from pprint import pformat
from pykalman import KalmanFilter

from Misc import sortKey, processArguments, drawBox

# def processArguments(args, params):
#     # arguments specified as 'arg_name=argv_val'
#     no_of_args = len(args)
#     for arg_id in range(no_of_args):
#         arg = args[arg_id].split('=')
#         if len(arg) != 2 or not arg[0] in params.keys():
#             print('Invalid argument provided: {:s}'.format(args[arg_id]))
#             return
#         if not arg[1] or not arg[0]:
#             continue
#         try:
#             params[arg[0]] = type(params[arg[0]])(arg[1])
#         except ValueError:
#             print('Invalid argument value {} provided for {}'.format(arg[1], arg[0]))
#             return


params = {
    'db_root_dir': '',
    'actor': '',
    'seq_name': '',
    'vid_fmt': '',
    'dst_dir': '',
    'show_img': 0,
    'n_frames': 0,
    'evenly_spaced': 0,
    'crop': 0,
    'reverse': 0,
    'roi': [],
    'resize_factor': 1.0,
    'start_id': 0,
    'out_fname_templ': 'image%06d',
    'ext': 'jpg',
    'mode': 0,
    'recursive': 1,
    'tracker_type': 0,
    'kalman_filtering': 0,
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)

    db_root_dir = params['db_root_dir']
    actor = params['actor']
    _seq_name = params['seq_name']
    show_img = params['show_img']
    vid_fmt = params['vid_fmt']
    n_frames = params['n_frames']
    evenly_spaced = params['evenly_spaced']
    roi = params['roi']
    resize_factor = params['resize_factor']
    dst_dir = params['dst_dir']
    start_id = params['start_id']
    out_fname_templ = params['out_fname_templ']
    crop = params['crop']
    reverse = params['reverse']
    ext = params['ext']
    mode = params['mode']
    recursive = params['recursive']
    tracker_type = params['tracker_type']
    kalman_filtering = params['kalman_filtering']

    vid_exts = ['.mkv', '.mp4', '.avi', '.mjpg', '.wmv', '.gif']
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

    roi_enabled = False
    if roi and isinstance(roi, (list, tuple)) and len(roi) == 4:
        xmin, ymin, xmax, ymax = roi
        if xmax > xmin and ymax > ymin:
            print('Using roi: ', roi)
            roi_enabled = True

    # _seq_name = os.path.abspath(_seq_name)

    if os.path.isdir(_seq_name):
        if mode == 0:
            print('Looking for source videos in: {}'.format(_seq_name))
            if recursive:
                print('searching recursively')
                video_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                                   os.path.splitext(f.lower())[1] in vid_exts]
                                  for (dirpath, dirnames, filenames) in os.walk(_seq_name, followlinks=True)]
                seq_names = [item for sublist in video_file_gen for item in sublist]
            else:
                seq_names = [os.path.join(_seq_name, k) for k in os.listdir(_seq_name) for _ext in vid_exts if
                             k.endswith(_ext)]
        else:
            print('Looking for source image sequences in: {}'.format(_seq_name))
            if recursive:
                print('searching recursively')
                video_file_gen = [[os.path.join(dirpath, d) for d in dirnames if
                                   any([os.path.splitext(f.lower())[1] in img_exts
                                        for f in os.listdir(os.path.join(dirpath, d))])]
                                  for (dirpath, dirnames, filenames) in os.walk(_seq_name, followlinks=True)]
                seq_names = [item for sublist in video_file_gen for item in sublist]
            else:
                seq_names = [os.path.join(_seq_name, k) for k in os.listdir(_seq_name) if
                             any([os.path.splitext(f.lower())[1] in img_exts
                                  for f in os.listdir(os.path.join(_seq_name, k))
                                  ])]
            if not seq_names:
                seq_names = [_seq_name]

        n_videos = len(seq_names)
        if n_videos <= 0:
            raise SystemError('No input videos found')

        print('n_videos: {}'.format(n_videos))
        seq_names.sort(key=sortKey)
    else:
        seq_names = [_seq_name]

    if reverse:
        print('Reversing videos')

    for seq_name in seq_names:
        src_path = seq_name
        if vid_fmt:
            src_path = src_path + '.' + vid_fmt
        if actor:
            print('actor: ', actor)
            src_path = os.path.join(actor, src_path)
        if db_root_dir:
            print('db_root_dir: ', db_root_dir)
            src_path = os.path.join(db_root_dir, src_path)

        if mode == 0 and not os.path.isfile(src_path):
            raise IOError('Invalid video file: {}'.format(src_path))

        print('seq_name: ', seq_name)
        print('start_id: ', start_id)
        print('Reading video file: {:s}'.format(src_path))

        if not dst_dir:
            out_seq_name = os.path.splitext(os.path.basename(src_path))[0]
            if crop:
                out_seq_name = '{}_crop'.format(out_seq_name)
                if tracker_type:
                    out_seq_name = '{}_track_{}'.format(out_seq_name, tracker_type)
            elif roi_enabled:
                out_seq_name = '{}_roi_{}_{}_{}_{}'.format(out_seq_name, xmin, ymin, xmax, ymax)

            dst_dir = os.path.join(os.path.dirname(src_path), out_seq_name)

        if dst_dir == src_path:
            raise IOError('Source and destination paths are identical')

        if dst_dir and not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        print('Writing image sequence to: {:s}/{:s}.{}'.format(dst_dir, out_fname_templ, ext))
        _src_files = []

        _ext = os.path.splitext(src_path)[1]
        if mode == 0:
            if _ext == '.gif':
                gif = imageio.mimread(src_path)
                meta_data = [img.meta for img in gif]
                print('gif meta_data: {}'.format(pformat(meta_data)))
                _src_files = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3
                              else cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                              for img in gif]
                total_frames = len(_src_files)
            else:
                cap = cv2.VideoCapture()
                if not cap.open(src_path):
                    raise SystemError('The video file ' + src_path + ' could not be opened')

                if cv2.__version__.startswith('2'):
                    cv_prop = cv2.cv.CAP_PROP_FRAME_COUNT
                else:
                    cv_prop = cv2.CAP_PROP_FRAME_COUNT

                total_frames = int(cap.get(cv_prop))
        else:
            _src_files = [os.path.join(src_path, k) for k in os.listdir(src_path) for _ext in img_exts if
                          k.endswith(_ext)]
            total_frames = len(_src_files)
            if total_frames <= 0:
                raise SystemError('No input frames found')
            _src_files.sort(key=sortKey)
            print('total_frames: {}'.format(total_frames))

        frame_gap = 1

        if n_frames <= 0:
            n_frames = total_frames
        else:
            if total_frames > 0 and n_frames > total_frames:
                raise AssertionError('Invalid n_frames {} for video with {} frames'.format(n_frames, total_frames))
            if evenly_spaced:
                frame_gap = total_frames / n_frames
                print('Using evenly spaced sampling with frame_gap: {} to sample {} frames'.format(
                    frame_gap, n_frames
                ))

        frame_id = all_frame_id = 0
        print_diff = max(1, int(n_frames / 100))
        start_t = time.time()
        means = [[0, 0], ]
        covariances = [[0, 0], ]
        tracker = None
        if tracker_type:
            _pause = 1
        else:
            _pause = 0
        measurements = []
        while True:
            if _src_files:
                frame = _src_files[frame_id]
                if isinstance(frame, str):
                    frame = cv2.imread(frame)
            else:
                ret, frame = cap.read()
                if not ret:
                    print('\nFrame {:d} could not be read'.format(frame_id + 1))
                    break
            if crop and frame_id == 0:
                roi = cv2.selectROI('Select ROI', frame)
                print('roi: {}'.format(roi))
                cv2.destroyWindow('Select ROI')
                x1, y1, w, h = roi

                if w == 0 or h == 0:
                    sys.exit(0)

                print('Using roi: ', roi)
                roi_enabled = True

                if crop == 2:
                    y1 = 0
                    h = frame.shape[1]

                roi = x1, y1, x1 + w, y1 + h

                if tracker_type:
                    track_roi = cv2.selectROI('Select object to track', frame)
                    cv2.destroyWindow('Select object to track')
                    track_x, track_y, tw, th = track_roi

                    if tw == 0 or th == 0:
                        track_roi = [x1, y1, w, h]

                    track_x, track_y, tw, th = track_roi

                    print('track_roi: {}'.format(track_roi))

                    if tracker_type == 1:
                        from siamfc.SiamFC import SiamFC

                        tracker = SiamFC()
                        print('Using SiamFC tracker')

                    elif tracker_type == 2:
                        from DaSiamRPN.DaSiamRPN import DaSiamRPN

                        tracker = DaSiamRPN()
                        print('Using DaSiamRPN tracker')

                    cx = track_x + tw / 2.0
                    cy = track_y + th / 2.0
                    bbox = [cx, cy, tw, th]
                    tracker.initialize(frame, bbox)
                    if kalman_filtering:
                        kf = KalmanFilter(n_dim_obs=2, n_dim_state=2)

                    if show_img:
                        frame_disp = np.copy(frame)
                        drawBox(frame_disp, track_x, track_y, track_x + tw, track_y + th)
                        cv2.imshow('frame_disp', frame_disp)

            elif crop and tracker_type:
                bbox = tracker.update(frame)
                _track_x, _track_y, _track_w, _track_h = bbox

                if show_img:
                    frame_disp = np.copy(frame)
                    drawBox(frame_disp, _track_x, _track_y, _track_x + _track_w, _track_y + _track_h)
                    cv2.imshow('frame_disp', frame_disp)

                tx, ty = _track_x - track_x, _track_y - track_y
                measurements.append([tx, ty])

                if frame_id > 1:
                    tx, ty = np.mean(measurements, axis=0)

                if kalman_filtering:
                    measurements.append([tx, ty])
                    # measurements = [[tx, ty], ]
                    # means, covariances = kf.filter_update(
                    #     means[-1], covariances[-1], measurements)
                    # print('\nmeasurements: {}'.format(measurements))
                    # print('\nmeans: {}'.format(means))
                    # # print('\ncovariances: {}'.format(covariances))

                    if frame_id == 1:
                        _measurements = np.asarray(measurements)
                        print('\nmeasurements: {}'.format(_measurements))
                        print('\nmeasurements.shape: {}'.format(_measurements.shape))

                        means, covariances = kf.filter(_measurements)

                        print('\nmeans:\n{}'.format(means))
                    elif frame_id > 1:
                        measurements = [[tx, ty], ]
                        _measurements = np.asarray(measurements)

                        means, covariances = kf.filter_update(
                            means[-1], covariances[-1], _measurements)
                        print('\nmeans:\n{}'.format(means))
                        # print('\ncovariances: {}'.format(covariances))

                        # print('\nnext_mean: {}'.format(next_mean))
                        # print('\nnext_covariance: {}'.format(next_covariance))
                        tx, ty = means[-1]

                x1, y1, x2, y2 = roi

                x1 += tx
                x2 += tx
                if tx > 0 and x2 > frame.shape[1]:
                    x1 -= (x2 - frame.shape[1])
                    x2 = frame.shape[1]
                if tx < 0 and x1 < 0:
                    x2 += x1
                    x1 = 0
                y1 += ty
                y2 += ty
                if ty > 0 and y2 > frame.shape[0]:
                    y1 -= (y2 - frame.shape[0])
                    y2 = frame.shape[0]
                if ty < 0 and y1 < 0:
                    y2 += y1
                    y1 = 0

                roi = [int(x1), int(y1), int(x2), int(y2)]

                track_x = _track_x
                track_y = _track_y

            all_frame_id += 1

            if frame_gap > 1 and all_frame_id % frame_gap != 0:
                continue

            frame_id += 1

            if frame_id <= start_id:
                sys.stdout.write('\rSkipped {:d}/{:d} frames'.format(
                    frame_id, start_id))
                sys.stdout.flush()
                continue
            if roi_enabled:
                frame = frame[roi[1]:roi[3], roi[0]:roi[2], :]
            if resize_factor != 1:
                frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

            out_id = (frame_id - start_id)
            if reverse:
                out_id = total_frames - out_id + 1
            out_path = os.path.join(dst_dir, out_fname_templ % out_id + '.' + ext)
            curr_img = cv2.imwrite(out_path, frame)
            if show_img:
                cv2.imshow('Frame', frame)
                _k = cv2.waitKey(1 - _pause)
                if _k == 27:
                    break
                elif _k == 32:
                    _pause = 1 - _pause
            if n_frames > 0 and (frame_id - start_id) >= n_frames:
                break

            if frame_id % print_diff == 0:
                end_t = time.time()
                fps = float(print_diff) / (end_t - start_t)
                sys.stdout.write('\rDone {:d}/{:d} frames at {:.4f} fps'.format(
                    (frame_id - start_id), n_frames, fps))
                sys.stdout.flush()
                start_t = end_t

        sys.stdout.write('\n\n')
        sys.stdout.flush()
        dst_dir = ''
        n_frames = 0
