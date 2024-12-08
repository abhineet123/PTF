import os
import shutil
import sys
from pprint import pformat
import cv2

import paramparse

from Misc import sortKey, resizeAR


class Params:
    videos = ['.', ]
    fps = 30


def main():
    params: Params = paramparse.process(Params)

    videos = params.videos
    fps = params.fps

    vid_exts = ['.mkv', '.mp4', '.avi', '.mjpg', '.wmv', '.gif', '.webm']

    if len(videos) == 1:
        videos = [os.path.join(videos[0], k) for k in os.listdir(videos[0]) for _ext in vid_exts if
                  k.endswith(_ext)]

    videos.sort(key=sortKey)

    print('merging videos {}'.format(videos))



    ext_to_codec = {
        '.avi': 'XVID',
        '.mkv': 'H264',
        '.mp4': 'H264',
        '.webm': 'XVID',
    }
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

if __name__ == '__main__':
    main()
