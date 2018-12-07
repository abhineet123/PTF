import sys
import os
import cv2

from Misc import processArguments


def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def check_for_similar_images(_filename, paths, methodName="Hellinger", n_results=10):
    valid_exts = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']

    OPENCV_METHODS = (
        ("Correlation", cv2.HISTCMP_CORREL),
        ("Chi-Squared", cv2.HISTCMP_CHISQR),
        ("Intersection", cv2.HISTCMP_INTERSECT),
        ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))

    print('Looking for images similar to {} in {}'.format(_filename, paths))

    results = {}
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
    print('img_hist: {}'.format(img_hist))

    n_files = n_skips = 0

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

                curr_image = cv2.imread(full_path)
                curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
                curr_hist = cv2.calcHist([curr_image], [0, 1, 2], None, [8, 8, 8],
                                         [0, 256, 0, 256, 0, 256])
                cv2.normalize(curr_hist, curr_hist).flatten()

                d = cv2.compareHist(img_hist, curr_hist, method)
                results[full_path] = d

                n_files += 1
                sys.stdout.write('\rSearched {} files'.format(n_files))
                sys.stdout.flush()
    results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
    print('\nTotal files searched: {}'.format(n_files))

    print('Similar files found: ')
    for (i, (v, k)) in enumerate(results):
        print('{} :: {}'.format(k, v))
        curr_image = cv2.imread(k)
        cv2.imshow('curr_image', curr_image)
        k = cv2.waitKey(0)
        if k == 27 or i >= n_results:
            break

    print('Files skipped: {}'.format(n_skips))


if __name__ == "__main__":
    params = {
        'filename': '',
        'root_dir': '.',
        'delete_file': 0,
    }
    processArguments(sys.argv[1:], params)
    filename = params['filename']
    root_dir = params['root_dir']
    delete_file = params['delete_file']

    check_for_similar_images(filename, root_dir)
