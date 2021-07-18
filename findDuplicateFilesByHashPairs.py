import sys
import os
import hashlib
from pprint import pprint
import itertools, inspect

import numpy as np
import cv2

try:
    import cPickle as pickle
except:
    import pickle

from Misc import processArguments


def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def get_hash(file_path):
    hashobj = hashlib.sha1()
    for chunk in chunk_reader(open(file_path, 'rb')):
        hashobj.update(chunk)
    file_id = (hashobj.digest(), os.path.getsize(file_path))
    return file_id


def main():
    params = {
        'files': '',
        'root_dir': '.',
        'delete_file': 0,
        'db_file': '',
        'file_type': '',
        'show_img': 0,
    }
    processArguments(sys.argv[1:], params)
    files = params['files']
    root_dir = params['root_dir']
    delete_file = params['delete_file']
    db_file = params['db_file']
    file_type = params['file_type']
    show_img = params['show_img']

    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    vid_exts = ['.mp4', '.mkv', '.avi', '.wmv', '.3gp', '.webm', '.mpeg', '.mjpg']

    if file_type == 'img':
        print('Only searching for images')
        valid_exts = img_exts
    else:
        show_img = 0
        if file_type == 'vid':
            print('Only searching for videos')
            valid_exts = vid_exts
        elif file_type:
            valid_exts = ['.' + file_type, ]
        else:
            valid_exts = None

    src_file_gen = [[os.path.abspath(os.path.join(dirpath, f)) for f in files]
                    for (dirpath, dirnames, files) in os.walk(root_dir, followlinks=True)]
    src_file_list = [item for sublist in src_file_gen for item in sublist]

    if valid_exts:
        print('Looking only for files with ext: {}'.format(valid_exts))
        src_file_list = [k for k in src_file_list if os.path.splitext(os.path.basename(k).lower())[1] in valid_exts]

    n_files = len(src_file_list)
    print('n_files: {}'.format(n_files))

    all_stats = {k: os.stat(k) for k in src_file_list}
    # pprint(all_stats)

    all_stats = {'{}_{}'.format(st.st_ino, st.st_dev): k for k, st in all_stats.items()}

    # pprint(all_stats)

    n_files = len(all_stats)

    script_filename = inspect.getframeinfo(inspect.currentframe()).filename
    script_path = os.path.dirname(os.path.abspath(script_filename))

    db = {}
    if db_file:
        db_file = os.path.join(script_path, 'log', db_file)

        if os.path.isfile(db_file):
            print('Loading file hashes from {}'.format(db_file))
            db = pickle.load(open(db_file, "rb"))
        else:
            db_file_dir = os.path.dirname(db_file)
            if not os.path.isdir(db_file_dir):
                os.makedirs(db_file_dir)

    new_stats = [k for k in all_stats if k not in db
                 or db[k][0] != os.path.getmtime(all_stats[k])]

    n_new_files = len(new_stats)

    if new_stats:
        print('Computing hashes for {}/{} files ...'.format(n_new_files, n_files))
        db.update({k: (os.path.getmtime(all_stats[k]), get_hash(all_stats[k]))
                   for k in new_stats})
    else:
        print('No new files to compute hashes for')
    # src_file_hash_list = list(src_file_hash_dict.keys())
    # src_file_hash_set = set(src_file_hash_list)
    _new_stats = []

    if files:
        if os.path.isfile(files):
            print('Looking for duplicates of {}'.format(files))
            file_hash = get_hash(files)
            duplicates = [(files, all_stats[k]) for k in all_stats if db[k][1] == file_hash]
        elif os.path.isdir(files):
            _src_file_gen = [[os.path.abspath(os.path.join(_dirpath, f)) for f in _filenames]
                             for (_dirpath, _dirnames, _filenames) in os.walk(files, followlinks=True)]
            _src_file_list = [item for sublist in _src_file_gen for item in sublist]
            if valid_exts:
                print('Looking only for orig files with ext: {}'.format(valid_exts))
                _src_file_list = [k for k in _src_file_list if
                                  os.path.splitext(os.path.basename(k).lower())[1] in valid_exts]

            _all_stats = {k: os.stat(k) for k in _src_file_list}
            # pprint(all_stats)

            _all_stats = {'{}_{}'.format(st.st_ino, st.st_dev): k for k, st in _all_stats.items()}
            _n_files = len(_all_stats)

            _new_stats = [k for k in _all_stats if k not in db
                          or db[k][0] != os.path.getmtime(_all_stats[k])]
            _n_new_files = len(_new_stats)

            if _new_stats:
                print('Computing hashes for {}/{} orig files ...'.format(_n_new_files, _n_files))
                db.update({k: (os.path.getmtime(_all_stats[k]), get_hash(_all_stats[k]))
                           for k in _new_stats})

            print('Looking for duplicates of {} files in {} among {} files in {}'.format(
                _n_files, files, n_files, root_dir))

            duplicates = [(_all_stats[k1], all_stats[k2]) for k1 in _all_stats for k2 in all_stats
                          if db[k1][1] == db[k2][1] and k2 not in _all_stats]
    else:
        print('Looking for duplicates...')
        duplicates = [(all_stats[k[0]], all_stats[k[1]])
                      for k in itertools.combinations(all_stats, r=2)
                      if db[k[0]][1] == db[k[1]][1]]
    # duplicates = []
    # for pair in itertools.combinations(src_file_hash_list, r=2):
    #     if pair[0][0] == pair[1][0]:
    #         duplicates.append((pair[0][1], pair[1][1]))
    if duplicates:
        n_duplicates = len(duplicates)
        print('Found {} duplicates:'.format(n_duplicates))
        pprint(duplicates)
        if delete_file or show_img:
            _pause = 1
            for pair in duplicates:
                if show_img:
                    vis_img = np.concatenate(
                        (cv2.imread(pair[0]), cv2.imread(pair[1])), axis=1)
                    cv2.imshow('duplicates', vis_img)
                    k = cv2.waitKey(1 - _pause)
                    if k == 32:
                        _pause = 1 - _pause
                    elif k == 27:
                        break

                if delete_file:
                    del_path = pair[delete_file - 1]
                    if os.path.isfile(del_path):
                        print('Deleting {}'.format(del_path))
                        os.remove(del_path)

            if delete_file:
                print('Deleted {} duplicates'.format(n_duplicates))

    else:
        print('No duplicates found')

    if db_file and (new_stats or _new_stats):
        print('Saving hash db to: {}'.format(db_file))
        pickle.dump(db, open(db_file, "wb"))


if __name__ == "__main__":
    main()
