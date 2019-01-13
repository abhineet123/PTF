import sys
import os
import hashlib
from pprint import pprint
import itertools

from Misc import processArguments


def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def getHash(file_path):
    hashobj = hashlib.sha1()
    for chunk in chunk_reader(open(file_path, 'rb')):
        hashobj.update(chunk)
    file_id = (hashobj.digest(), os.path.getsize(file_path))
    return file_id


if __name__ == "__main__":
    params = {
        'filename': '',
        'root_dir': '.',
        'delete_file': 0,
        'file_type': '',
    }
    processArguments(sys.argv[1:], params)
    filename = params['filename']
    root_dir = params['root_dir']
    delete_file = params['delete_file']
    file_type = params['file_type']

    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    vid_exts = ['.mp4', '.mkv', '.avi', '.wmv', '.3gp', '.webm', '.mpeg', '.mjpg']

    if file_type == 'img':
        print('Only searching for images')
        valid_exts = img_exts
    elif file_type == 'vid':
        print('Only searching for videos')
        valid_exts = vid_exts
    elif file_type:
        valid_exts = ['.' + file_type, ]
    else:
        valid_exts = None

    src_file_gen = [[os.path.join(dirpath, f) for f in filenames]
                    for (dirpath, dirnames, filenames) in os.walk(root_dir, followlinks=True)]
    src_file_list = [item for sublist in src_file_gen for item in sublist]

    if valid_exts:
        print('Looking only for files with ext: {}'.format(valid_exts))
        src_file_list = [k for k in src_file_list if os.path.splitext(os.path.basename(k).lower())[1] in valid_exts]

    n_files = len(src_file_list)

    print('n_files: {}'.format(n_files))

    print('Computing file hashes...')
    src_file_hash_list = [(getHash(k), k) for k in src_file_list]

    # src_file_hash_list = list(src_file_hash_dict.keys())
    # src_file_hash_set = set(src_file_hash_list)
    if filename:
        print('Looking for duplicates of {}'.format(filename))

        file_hash = getHash(filename)
        duplicates = [(filename, k[1]) for k in src_file_hash_list if k[0] == file_hash]
    else:
        print('Looking for duplicates...')
        duplicates = [(pair[0][1], pair[1][1]) for pair in itertools.combinations(src_file_hash_list, r=2)
                      if pair[0][0] == pair[1][0]]
    # duplicates = []
    # for pair in itertools.combinations(src_file_hash_list, r=2):
    #     if pair[0][0] == pair[1][0]:
    #         duplicates.append((pair[0][1], pair[1][1]))

    n_duplicates = len(duplicates)
    print('Found {} duplicates:'.format(n_duplicates))
    if n_duplicates > 0:
        pprint(duplicates)
        if delete_file:
            for pair in duplicates:
                del_path = pair[1]
                if os.path.isfile(del_path):
                    print('Deleting {}'.format(del_path))
                    os.remove(del_path)
