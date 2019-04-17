import os
import cv2
import sys

from Misc import sortKey, processArguments

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
    'reverse': 0,
    'roi': [],
    'resize_factor': 1.0,
    'start_id': 0,
    'out_fname_templ': 'image%06d',
    'ext': 'jpg'
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)

    db_root_dir = params['db_root_dir']
    actor = params['actor']
    _seq_name = params['seq_name']
    show_img = params['show_img']
    vid_fmt = params['vid_fmt']
    n_frames = params['n_frames']
    roi = params['roi']
    resize_factor = params['resize_factor']
    dst_dir = params['dst_dir']
    start_id = params['start_id']
    out_fname_templ = params['out_fname_templ']
    reverse = params['reverse']
    ext = params['ext']

    vid_exts = ['.mkv', '.mp4', '.avi', '.mjpg', '.wmv']

    roi_enabled = False
    if roi and isinstance(roi, (list, tuple)) and len(roi) == 4:
        xmin, ymin, xmax, ymax = roi
        if xmax > xmin and ymax > ymin:
            print('Using roi: ', roi)
            roi_enabled = True

    # _seq_name = os.path.abspath(_seq_name)

    if os.path.isdir(_seq_name):
        print('Reading source videos from: {}'.format(_seq_name))
        seq_names = [k for k in os.listdir(_seq_name) for _ext in vid_exts if k.endswith(_ext)]
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

        if not os.path.isfile(src_path):
            raise IOError('Invalid video file: {}'.format(src_path))

        print('seq_name: ', seq_name)
        print('start_id: ', start_id)
        print('Reading video file: {:s}'.format(src_path))

        if not dst_dir:
            out_seq_name = os.path.splitext(os.path.basename(src_path))[0]
            if roi_enabled:
                out_seq_name = '{}_roi_{}_{}_{}_{}'.format(out_seq_name, xmin, ymin, xmax, ymax)
            dst_dir = os.path.join(os.path.dirname(src_path), out_seq_name)
        if dst_dir and not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        print('Writing image sequence to: {:s}/{:s}.{}'.format(dst_dir, out_fname_templ, ext))

        cap = cv2.VideoCapture()
        if not cap.open(src_path):
            raise StandardError('The video file ' + src_path + ' could not be opened')

        if cv2.__version__.startswith('2'):
            cv_prop = cv2.cv.CAP_PROP_FRAME_COUNT
        else:
            cv_prop = cv2.CAP_PROP_FRAME_COUNT

        total_frames = int(cap.get(cv_prop))

        if n_frames <= 0:
            n_frames = total_frames
        elif total_frames > 0 and n_frames > total_frames:
            raise AssertionError('Invalid n_frames {} for video with {} frames'.format(n_frames, total_frames))

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print('\nFrame {:d} could not be read'.format(frame_id + 1))
                break
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
                if cv2.waitKey(1) == 27:
                    break
            if n_frames > 0 and (frame_id - start_id) >= n_frames:
                break
            sys.stdout.write('\rDone {:d}/{:d} frames'.format(
                (frame_id - start_id), n_frames))
            sys.stdout.flush()
        sys.stdout.write('\n\n')
        sys.stdout.flush()
        dst_dir = ''
        n_frames = 0
