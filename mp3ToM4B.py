import cv2
import sys
import os, shutil

from pprint import pformat
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

img_exts = ['.mp3', '.MP3']

if os.path.isdir(_src_path):
    src_files = [k for k in os.listdir(_src_path) for _ext in img_exts if k.endswith(_ext)]
    if not src_files:
        # src_paths = [os.path.join(_src_path, k) for k in os.listdir(_src_path) if
        #              os.path.isdir(os.path.join(_src_path, k))]
        video_file_gen = [[os.path.join(dirpath, d) for d in dirnames if
                           any([os.path.splitext(f.lower())[1] in img_exts
                                for f in os.listdir(os.path.join(dirpath, d))])]
                          for (dirpath, dirnames, filenames) in os.walk(_src_path, followlinks=True)]
        src_paths = [item for sublist in video_file_gen for item in sublist]
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

n_src_paths = len(src_paths)

for src_id, src_path in enumerate(src_paths):
    src_path = os.path.abspath(src_path)
    seq_name = os.path.basename(src_path)

    print('{}/{} Reading mp3 files from: {}'.format(src_id + 1, n_src_paths, src_path))

    src_path = os.path.abspath(src_path)
    src_files = [k for k in os.listdir(src_path) for _ext in img_exts if k.endswith(_ext)]
    n_src_files = len(src_files)
    if n_src_files <= 0:
        raise SystemError('No input frames found')
    src_files.sort(key=sortKey)
    print('n_src_files: {}'.format(n_src_files))

    src_files = ['{}'.format(os.path.join(src_path, k)) for k in src_files]

    src_files_str = '|'.join(src_files)

    dst_path = src_path.replace('-', '_').replace('-', '_')
    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)
    dst_seq_name = seq_name.replace('-', '_').replace('-', '_')

    out_path = os.path.join(dst_path, '{}.m4b'.format(dst_seq_name))

    cmd = 'ffmpeg -i "concat:{}" -c:a aac -strict experimental -b:a 64k -f mp4 "{}"'.format(src_files_str, out_path)
    print('Running {}'.format(cmd))

    os.system(cmd)
