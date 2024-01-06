import os
import cv2
import sys
import imageio
from pprint import pformat
import platform

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
    'seq_name': '.',
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
    evenly_spaced = params['evenly_spaced']
    roi = params['roi']
    resize_factor = params['resize_factor']
    dst_dir = params['dst_dir']
    start_id = params['start_id']
    out_fname_templ = params['out_fname_templ']
    crop = params['crop']
    reverse = params['reverse']
    ext = params['ext']

    zip_exts = ['.zip', '.tar.gz', '.tar']

    # _seq_name = os.path.abspath(_seq_name)

    _seq_name = os.path.abspath(_seq_name)

    if platform.system() == 'Windows':
        base_cmd = 'tar -xf'
    elif platform.system() == 'Linux':
        base_cmd = 'unzip'

    if os.path.isdir(_seq_name):
        print('Reading source archives from: {}'.format(_seq_name))
        video_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                           os.path.splitext(f.lower())[1] in zip_exts]
                          for (dirpath, dirnames, filenames) in os.walk(_seq_name, followlinks=True)]
        archives_list = [item for sublist in video_file_gen for item in sublist]

        n_archives = len(archives_list)
        if n_archives <= 0:
            raise SystemError('No input archives found')
        print('n_archives: {}'.format(n_archives))
        archives_list.sort(key=sortKey)
    else:
        archives_list = [_seq_name]

    for seq_name in archives_list:
        src_path = seq_name

        src_path = src_path.replace(os.sep, '/')

        src_dir = os.path.dirname(src_path)

        if src_dir == _seq_name:
            continue

        if not os.path.isfile(src_path):
            raise IOError('Invalid archive file: {}'.format(src_path))

        if platform.system() == 'Windows':
            cmd = 'tar -xf "{}" -C "{}"'.format(src_path, src_dir)
        elif platform.system() == 'Linux':
            cmd = 'cd "{}" && unzip "{}"'.format(src_dir, src_path)
        print('Running {}'.format(cmd))
        os.system(cmd)
