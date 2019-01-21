import sys
import os
import hashlib
from pprint import pprint
import itertools, inspect

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


def getHash(file_path):
    hashobj = hashlib.sha1()
    for chunk in chunk_reader(open(file_path, 'rb')):
        hashobj.update(chunk)
    file_id = (hashobj.digest(), os.path.getsize(file_path))
    return file_id


def main():
    params = {
        'filename': '',
        'root_dir': '.',
        'delete_file': 0,
        'db_file': '',
        'file_type': '',
    }
    processArguments(sys.argv[1:], params)
    filename = params['filename']
    root_dir = params['root_dir']
    delete_file = params['delete_file']
    db_file = params['db_file']
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

    src_file_gen = [[os.path.abspath(os.path.join(dirpath, f)) for f in filenames]
                    for (dirpath, dirnames, filenames) in os.walk(root_dir, followlinks=True)]
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
        db.update({k: (os.path.getmtime(all_stats[k]), getHash(all_stats[k]))
                   for k in new_stats})
    else:
        print('No new files to compute hashes for')
    # src_file_hash_list = list(src_file_hash_dict.keys())
    # src_file_hash_set = set(src_file_hash_list)
    if filename:
        print('Looking for duplicates of {}'.format(filename))

        file_hash = getHash(filename)
        duplicates = [(filename, all_stats[k]) for k in all_stats if db[k][1] == file_hash]
    else:
        print('Looking for duplicates...')
        duplicates = [(all_stats[k[0]], all_stats[k[1]])
                      for k in itertools.combinations(all_stats, r=2)
                      if db[k[0]][1] == db[k[1]][1]]
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

    if db_file and new_stats:
        print('Saving hash db to: {}'.format(db_file))
        pickle.dump(db, open(db_file, "wb"))


if __name__ == "__main__":
    main()
