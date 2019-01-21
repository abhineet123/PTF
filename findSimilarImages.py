import sys, os, inspect, itertools
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


def getHist(full_path):
    curr_image = cv2.imread(full_path)
    curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
    curr_hist = cv2.calcHist([curr_image], [0, 1, 2], None, [8, 8, 8],
                             [0, 256, 0, 256, 0, 256])
    cv2.normalize(curr_hist, curr_hist).flatten()
    return curr_hist


def check_for_similar_images(_filename, paths, db_file, methodName="Hellinger", n_results=10, thresh=0.1):
    valid_exts = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']

    OPENCV_METHODS = (
        ("Correlation", cv2.HISTCMP_CORREL),
        ("Chi-Squared", cv2.HISTCMP_CHISQR),
        ("Intersection", cv2.HISTCMP_INTERSECT),
        ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))

    # results = {}
    reverse = False
    if methodName in ("Correlation", "Intersection"):
        reverse = True
    # method = OPENCV_METHODS[methodName]
    method = cv2.HISTCMP_BHATTACHARYYA

    script_filename = inspect.getframeinfo(inspect.currentframe()).filename
    script_path = os.path.dirname(os.path.abspath(script_filename))

    db = {}
    if db_file:
        db_file = os.path.join(script_path, 'log', db_file)

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

    # pprint(all_files_list)

    all_stats = {k: os.stat(k) for k in all_files_list}
    all_stats = {'{}_{}'.format(st.st_ino, st.st_dev): k for k, st in all_stats.items()}

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

    new_stats = [k for k in all_stats if k not in db
                 or db[k][0] != os.path.getmtime(all_stats[k])]

    n_new_files = len(new_stats)
    n_files = len(all_files_list)

    # if n_new_files:
    #     print('Computing features for {} / {} files'.format(n_new_files, len(all_files_list)))
    # else:
    #     print('No new files to compute features for')

    if new_stats:
        print('Computing hashes for {}/{} files ...'.format(n_new_files, n_files))
        db.update({k: (os.path.getmtime(all_stats[k]), getHist(all_stats[k]))
                   for k in new_stats})

    # for _stsats in new_stats:
    #     full_path = all_stats[_stsats]
    #
    #     db[_stsats] = (os.path.getmtime(full_path), curr_hist)
    #
    #     # d = cv2.compareHist(img_hist, curr_hist, method)
    #     # results[full_path] = d
    #
    #     # file_path_map[full_path] = full_path
    #
    #     n_files += 1
    #     sys.stdout.write('\rDone {} files'.format(n_files))
    #     sys.stdout.flush()

    print()

    if _filename:
        print('Looking for images similar to {} in {}'.format(_filename, paths))

        img_hist = getHist(_filename)

        # image = cv2.imread(_filename)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
        #                         [0, 256, 0, 256, 0, 256])
        # cv2.normalize(img_hist, img_hist).flatten()
        # print('img_hist: {}'.format(img_hist))

        print('Comparing features...')

        results = {all_stats[k]: cv2.compareHist(img_hist, db[k][1], method) for k in all_stats}
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
    else:
        # pairwise search
        print('Looking for pairwise similar images using thresh: {}'.format(thresh))
        print('Comparing features...')
        all_files_features_pairs = [(all_stats[k[0]], all_stats[k[1]],
                                     cv2.compareHist(db[k[0]][1], db[k[1]][1], method)) for k in
                                    itertools.combinations(list(all_stats.keys()), r=2)]
        print('Thresholding...')
        similar_img_pairs = [k for k in all_files_features_pairs if k[2] < thresh]
        print('Sorting...')
        similar_img_pairs.sort(key=lambda x: x[2])

        n_pairs = len(similar_img_pairs)
        cwd = os.getcwd()
        print('Found {} pairs'.format(n_pairs))
        for pair in similar_img_pairs:
            if not os.path.isfile(pair[0]):
                print('{} does not exist'.format(pair[0]))
                continue
            if not os.path.isfile(pair[1]):
                print('{} does not exist'.format(pair[1]))
                continue

            curr_image_1 = cv2.imread(pair[0])
            curr_image_2 = cv2.imread(pair[1])

            h1, w1 = curr_image_1.shape[:2]
            h2, w2 = curr_image_2.shape[:2]

            print('{}({}, {} x {})\t{}({}, {} x {})\t{}'.format(
                os.path.relpath(pair[0], cwd), os.path.getsize(pair[0]) / 1000.0, w1, h1,
                os.path.relpath(pair[1], cwd), os.path.getsize(pair[1]) / 1000.0, w2, h2,
                pair[2]))
            cv2.imshow('curr_image_1', curr_image_1)
            cv2.imshow('curr_image_2', curr_image_2)
            k = cv2.waitKey(0)
            if k == 27:
                break
            elif k == ord('1'):
                print('Deleting {}'.format(pair[0]))
                os.remove(pair[0])
            elif k == ord('2'):
                print('Deleting {}'.format(pair[1]))
                os.remove(pair[1])

    # print('Files skipped: {}'.format(n_skips))
    if db_file and new_stats:
        print('Saving feature db to: {}'.format(db_file))
        pickle.dump(db, open(db_file, "wb"))


def main():
    params = {
        'filename': '',
        'root_dir': '.',
        'delete_file': 0,
        'db_file': '',
        'thresh': 0.1,
    }

    processArguments(sys.argv[1:], params)
    filename = params['filename']
    root_dir = params['root_dir']
    delete_file = params['delete_file']
    db_file = params['db_file']
    thresh = params['thresh']

    check_for_similar_images(filename, root_dir, db_file, thresh=thresh)


if __name__ == "__main__":
    main()
