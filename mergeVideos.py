import os
import shutil
import sys
from pprint import pformat
import cv2

from Misc import processArguments, sortKey, resizeAR

# params = {
#     'dst_path': '.',
#     'file_ext': '',
#     'out_file': 'mis_log.txt',
#     'folder_name': '.',
#     'prefix': '',
#     'rename': 1,
#     'include_folders': 0,
#     'exceptions': [],
# }

videos = sys.argv[1:]

videos.sort(key=sortKey)

print('merging videos {}'.format(videos))

fps = 30

ext_to_codec = {
    '.avi': 'XVID',
    '.mkv': 'H264',
    '.mp4': 'H264',
    '.webm': 'XVID',
}
#
# processArguments(sys.argv[1:], params)
# dst_path = params['dst_path']
# file_ext = params['file_ext']
# out_file = params['out_file']
# folder_name = params['folder_name']
# prefix = params['prefix']
# include_folders = params['include_folders']
# exceptions = params['exceptions']
# rename = params['rename']
#
# dst_path = os.path.abspath(dst_path)
#
# img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff')
#
# if file_ext:
#     img_exts = [file_ext, ]
#
# if os.path.isfile(folder_name):
#     root_dir = os.path.abspath(os.getcwd())
#     subfolders = [x.strip() for x in open(folder_name).readlines()]
#     print('Looking for files with extension {:s} in sub folders of {:s} listed in {}'.format(
#         file_ext, root_dir, folder_name))
#     folder_name = root_dir
# else:
#     folder_name = os.path.abspath(folder_name)
#     print('Looking for files with extensions in {} in sub folders of {:s}'.format(img_exts, folder_name))
#     subfolders = [name for name in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, name))]
# if prefix:
#     print('Limiting search to only sub folders starting with {}'.format(prefix))
#     subfolders = [x for x in subfolders if x.startswith(prefix)]
# try:
#     subfolders.sort(key=sortKey)
# except:
#     subfolders.sort()
#
# if include_folders == 1:
#     print('Searching for folders too')
# elif include_folders == 2:
#     print('Searching only for folders')
# else:
#     print('Not searching for folders')
#
# if file_ext == '__n__':
#     file_ext = ''

total_files = 0
# out_fid = open(out_file, 'w')
files = []
empty_folders = []

video_out = None

for i, video in enumerate(videos):

    print('Reading video {}'.format(video))

    cap = cv2.VideoCapture(video)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_fname, video_ext = os.path.splitext(os.path.basename(video))

    if i == 0:
        video_dir = os.path.dirname(video)
        if video_ext in ext_to_codec.keys():
            out_video_ext = video_ext
            codec = ext_to_codec[video_ext]
        else:
            out_video_ext = '.avi'
            codec = 'XVID'

        out_video = os.path.join(video_dir, video_fname + '_merged' + out_video_ext)
        out_dir = os.path.join(video_dir, video_fname + '_merged')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_out = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
        if video_out is None:
            raise IOError('Output video file could not be opened: {}'.format(
                out_video))
        print('Writing video {}'.format(out_video))

    while True:
        ret, image = cap.read()
        if not ret:
            # print('frame {} could not be read from {}'.format(i, video))
            break

        _h, _w = image.shape[:2]
        if (_h, _w) != (h, w):
            print('resizing from {} to {}'.format((_h, _w), (h, w)))
            image = resizeAR(image, w, h)

        video_out.write(image)

    cap.release()

    video_dst = os.path.join(out_dir, video_fname + video_ext)
    print('moving {} to {}'.format(video, video_dst))
    shutil.move(video, video_dst)

video_out.release()
