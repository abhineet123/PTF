import sys
import os
import hashlib

from Misc import processArguments


def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def check_for_duplicates(paths, _delete_file, _file_type, _filename='', hash=hashlib.sha1):

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
    hashes = {}
    if _filename:
        print('Looking for duplicates of {} in {}'.format(_filename, paths))
        hashobj = hash()
        for chunk in chunk_reader(open(_filename, 'rb')):
            hashobj.update(chunk)
        file_id = (hashobj.digest(), os.path.getsize(_filename))
        hashes[file_id] = _filename

    n_duplicates = n_files = n_skips = 0

    for path in paths:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if valid_exts is not None:
                    file_ext = os.path.splitext(os.path.basename(filename))[1][1:]
                    if not file_ext.lower() in valid_exts:
                        print('\nSkipping {} with ext {}'.format(filename, file_ext))
                        n_skips += 1
                        continue
                full_path = os.path.join(dirpath, filename)
                hashobj = hash()
                for chunk in chunk_reader(open(full_path, 'rb')):
                    hashobj.update(chunk)
                file_id = (hashobj.digest(), os.path.getsize(full_path))
                duplicate = hashes.get(file_id, None)
                if duplicate:
                    if _filename:
                        print("\nDuplicate found: {}".format(full_path))
                        del_path = full_path
                    else:
                        print("\nDuplicate found: {} and {}".format(full_path, duplicate))
                        del_path = duplicate
                    n_duplicates += 1
                    if os.path.isfile(del_path) and _delete_file:
                        print('Deleting {}'.format(del_path))
                        os.remove(del_path)
                elif not _filename:
                    hashes[file_id] = full_path
                n_files += 1
                sys.stdout.write('\rSearched {} files'.format(n_files))
                sys.stdout.flush()
    print('\nTotal files searched: {}'.format(n_files))
    print('Duplicate files found: {}'.format(n_duplicates))
    print('Files skipped: {}'.format(n_skips))


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

    check_for_duplicates(root_dir, delete_file, file_type, filename)
