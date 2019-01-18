import sys, os, inspect
import cv2
from pprint import pprint

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


def check_for_similar_images(_filename, paths, db_file, methodName="Hellinger", n_results=10):
    valid_exts = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']

    OPENCV_METHODS = (
        ("Correlation", cv2.HISTCMP_CORREL),
        ("Chi-Squared", cv2.HISTCMP_CHISQR),
        ("Intersection", cv2.HISTCMP_INTERSECT),
        ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))

    print('Looking for images similar to {} in {}'.format(_filename, paths))

    # results = {}
    reverse = False
    if methodName in ("Correlation", "Intersection"):
        reverse = True
    # method = OPENCV_METHODS[methodName]
    method = cv2.HISTCMP_BHATTACHARYYA

    image = cv2.imread(_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
    cv2.normalize(img_hist, img_hist).flatten()
    # print('img_hist: {}'.format(img_hist))

    n_files = 0

    db = {}
    if db_file:
        if os.path.isfile(db_file):
            print('Loading feature  db from {}'.format(db_file))
            db = pickle.load(open(db_file, "rb"))
        else:
            db_file_dir = os.path.dirname(db_file)
            if not os.path.isdir(db_file_dir):
                os.makedirs(db_file_dir)
    if valid_exts:
        print('Looking only for files with ext: {}'.format(valid_exts))

    all_files_list = []
    for path in paths:
        _path = os.path.abspath(path)
        src_file_gen = [[os.path.join(dirpath, f) for f in filenames]
                        for (dirpath, dirnames, filenames) in os.walk(_path, followlinks=True)]
        src_file_list = [item for sublist in src_file_gen for item in sublist]

        if valid_exts:
            src_file_list = [k for k in src_file_list if
                             os.path.splitext(os.path.basename(k).lower())[1][1:] in valid_exts]
        all_files_list += src_file_list

    n_all_files = len(all_files_list)
    print('Searching {} files in all'.format(n_all_files))

    pprint(all_files_list)

    # print('Looking for existing files in db')
    # n_files = 0
    # file_path_map = {}
    # new_files_list = []
    # for _file in all_files_list:
    #     same_files = [k for k in db if os.path.samefile(_file, k)]
    #     if same_files:
    #         file_path_map[_file] = same_files[0]
    #     else:
    #         new_files_list.append(_file)
    #
    #     n_files += 1
    #     sys.stdout.write('\rDone {} files'.format(n_files))
    #     sys.stdout.flush()
    #
    # print()

    new_files_list = [k for k in all_files_list if k not in db or db[k][0] != os.path.getmtime(k)]

    n_new_files = len(new_files_list)
    if n_new_files:
        print('Computing features for {} / {} files'.format(n_new_files, len(all_files_list)))
    else:
        print('No new files to compute features for')

    for full_path in new_files_list:
        curr_image = cv2.imread(full_path)
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
        curr_hist = cv2.calcHist([curr_image], [0, 1, 2], None, [8, 8, 8],
                                 [0, 256, 0, 256, 0, 256])
        cv2.normalize(curr_hist, curr_hist).flatten()

        db[full_path] = (os.path.getmtime(full_path), curr_hist)

        # d = cv2.compareHist(img_hist, curr_hist, method)
        # results[full_path] = d

        # file_path_map[full_path] = full_path

        n_files += 1
        sys.stdout.write('\rDone {} files'.format(n_files))
        sys.stdout.flush()

    print('Comparing features...')
    results = {k: cv2.compareHist(img_hist, db[k][1], method) for k in all_files_list}
    results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
    # print('\nTotal files searched: {}'.format(n_files))

    print('Similar files found: ')
    for (i, (v, k)) in enumerate(results):
        print('{} :: {}'.format(k, v))
        curr_image = cv2.imread(k)
        cv2.imshow('curr_image', curr_image)
        k = cv2.waitKey(0)
        if k == 27 or i >= n_results:
            break

    # print('Files skipped: {}'.format(n_skips))
    if db_file and new_files_list:
        print('Saving feature db to: {}'.format(db_file))
        pickle.dump(db, open(db_file, "wb"))


if __name__ == "__main__":
    params = {
        'filename': '',
        'root_dir': '.',
        'delete_file': 0,
        'db_file': '',
    }

    processArguments(sys.argv[1:], params)
    filename = params['filename']
    root_dir = params['root_dir']
    delete_file = params['delete_file']
    db_file = params['db_file']

    if db_file:
        script_filename = inspect.getframeinfo(inspect.currentframe()).filename
        script_path = os.path.dirname(os.path.abspath(script_filename))
        db_file = os.path.join(script_path, 'log', db_file)

    check_for_similar_images(filename, root_dir, db_file)
