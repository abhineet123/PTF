import os
import paramparse
from tqdm import tqdm


from Misc import sortKey, drawBox, trim

class Params:
    def __init__(self):
        self.cfg = ()
        self.seq_name = '.'
        self.recursive = 1
        self.audio_ext = 'aac'


if __name__ == '__main__':
    params = Params()

    paramparse.process(params)
    _seq_name = params.seq_name
    audio_ext = params.audio_ext
    recursive = params.recursive

    vid_exts = ['.mkv', '.mp4', '.avi', '.mjpg', '.wmv', '.gif', '.webm']

    # _seq_name = os.path.abspath(_seq_name)

    if os.path.isdir(_seq_name):
        print('Looking for source videos in: {}'.format(_seq_name))

        if recursive:
            print('searching recursively')
            video_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                               os.path.splitext(f.lower())[1] in vid_exts]
                              for (dirpath, dirnames, filenames) in os.walk(_seq_name, followlinks=True)]
            vid_files = [item for sublist in video_file_gen for item in sublist]
        else:
            vid_files = [os.path.join(_seq_name, k) for k in os.listdir(_seq_name) for _ext in vid_exts if
                         k.endswith(_ext)]
    else:
        vid_files = [_seq_name, ]

    n_videos = len(vid_files)
    if n_videos <= 0:
        raise SystemError('No input videos found')

    print('n_videos: {}'.format(n_videos))
    vid_files.sort(key=sortKey)

    n_vid_files = len(vid_files)
    pbar = tqdm(vid_files)

    for vid_file in pbar:
        filename = os.path.splitext(os.path.basename(vid_file))[0]
        pbar.set_description(filename)
        ffmpeg_cmd = f'ffmpeg -hide_banner -loglevel error -i "{vid_file}" -vn -acodec copy "{filename}.{audio_ext}"'
        # print(f'{ffmpeg_cmd}')
        os.system(ffmpeg_cmd)

