import sys
import os
import shutil

from tqdm import tqdm

from Misc import processArguments


def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def check_for_duplicates(src_dir, dst_dir, mv_dir, _delete_file, _file_type):

    img_exts = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    vid_exts = ['mp4', 'mkv', 'avi', 'wmv', '3gp', 'webm', 'mpeg', 'mjpg']

    if _file_type == 'img':
        print('Only searching for images')
        valid_exts = img_exts
    elif _file_type == 'vid':
        print('Only searching for videos')
        valid_exts = vid_exts
    else:
        valid_exts = None

    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)

    print('Looking for source files in {}'.format(src_dir))

    if valid_exts:
        src_files_gen = [[os.path.join(dirpath, f) for f in filenames if
                           os.path.splitext(f.lower())[1] in valid_exts]
                          for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)]
    else:
        src_files_gen = [[os.path.join(dirpath, f) for f in filenames]
                          for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)]

    src_files = [item for sublist in src_files_gen for item in sublist]
    n_files = len(src_files)
    print('Found {} source files'.format(n_files))

    print('Looking for duplicate files in {}'.format(dst_dir))
    dst_files = [_path.replace(src_dir, dst_dir) for _path in src_files]
    dup_dst_files = [_path for _path in dst_files if os.path.isfile(_path)]
    n_duplicates = len(dup_dst_files)
    print('Found {} duplicate files'.format(n_duplicates))

    if not mv_dir:
        mv_dir = os.path.join(dst_dir, 'duplicates')
        if not os.path.isdir(mv_dir):
            os.makedirs(mv_dir)

    print('Moving duplicate files to {}'.format(mv_dir))
    for dst_file in tqdm(dup_dst_files):
        mv_file = dst_file.replace(dst_dir, mv_dir)
        mv_root_dir = os.path.dirname(mv_file)
        if not os.path.isdir(mv_root_dir):
            os.makedirs(mv_root_dir)

        # print('{} -> {}'.format(dst_file, mv_file))

        shutil.move(dst_file, mv_file)







if __name__ == "__main__":
    params = {
        'src_dir': '',
        'dst_dir': '.',
        'mv_dir': '',
        'delete_file': 0,
        'file_type': '',
    }
    processArguments(sys.argv[1:], params)
    src_dir = params['src_dir']
    dst_dir = params['dst_dir']
    mv_dir = params['mv_dir']
    delete_file = params['delete_file']
    file_type = params['file_type']

    check_for_duplicates(src_dir, dst_dir, mv_dir, delete_file, file_type)
