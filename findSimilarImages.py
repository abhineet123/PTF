import os
import inspect
import itertools
import numpy as np
import cv2
from pprint import pprint
from datetime import datetime
from tqdm import tqdm

import functools

import torch
from torch import nn
from tqdm import tqdm
from torchvision import models, transforms

try:
    import cPickle as pickle
except:
    import pickle

import paramparse

from Misc import resizeAR


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(VGG16FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def create_vgg16(crop=False):
    model = models.vgg16(pretrained=True)
    # model = models.resnet101(pretrained=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    new_model = VGG16FeatureExtractor(model)

    transform_list = [
        transforms.ToPILImage(),
    ]

    if crop:
        transform_list.append(transforms.CenterCrop(512))
        transform_list.append(transforms.Resize(448))

    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)

    return new_model, transform, device


def get_vgg16(img_path, model, transform, device):
    img_cv = cv2.imread(img_path)

    assert img_cv is not None, "invalid image path: {}".format(img_path)
    img_cv = resizeAR(img_cv, 448, 448)

    img = transform(img_cv)

    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)

    # img_disp = img.cpu().detach().numpy().squeeze().transpose([1, 2, 0])

    # cv2.imshow('img_cv', img_cv)
    # cv2.imshow('img_disp', img_disp)
    # cv2.waitKey(0)

    feature = model(img)

    feature = feature.cpu().detach().numpy().reshape(-1)

    return feature

def create_resnext101(crop=False):
    model = models.resnext101_32x8d(pretrained=True)
    # model = models.resnet101(pretrained=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    new_model = VGG16FeatureExtractor(model)

    transform_list = [
        transforms.ToPILImage(),
    ]

    if crop:
        transform_list.append(transforms.CenterCrop(512))
        transform_list.append(transforms.Resize(448))

    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)

    return new_model, transform, device


def get_resnext101(img_path, model, transform, device):
    img_cv = cv2.imread(img_path)

    assert img_cv is not None, "invalid image path: {}".format(img_path)
    img_cv = resizeAR(img_cv, 448, 448)

    img = transform(img_cv)

    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)

    # img_disp = img.cpu().detach().numpy().squeeze().transpose([1, 2, 0])

    # cv2.imshow('img_cv', img_cv)
    # cv2.imshow('img_disp', img_disp)
    # cv2.waitKey(0)

    feature = model(img)

    feature = feature.cpu().detach().numpy().reshape(-1)

    return feature


def euclidean_dist(a, b):
    return np.linalg.norm(a - b)


def get_hist(full_path, model=None):
    curr_image = cv2.imread(full_path)
    curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
    curr_hist = cv2.calcHist([curr_image], [0, 1, 2], None, [8, 8, 8],
                             [0, 256, 0, 256, 0, 256])
    cv2.normalize(curr_hist, curr_hist).flatten()
    return curr_hist


def check_for_similar_images(files, paths, db_file, feature_type, methodName="Hellinger", n_results=10, thresh=0.1):
    valid_exts = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    reverse = False

    if feature_type == 0:
        print('using normalized histogram as features')
        get_features = get_hist
        # results = {}
        OPENCV_METHODS = (
            ("Correlation", cv2.HISTCMP_CORREL),
            ("Chi-Squared", cv2.HISTCMP_CHISQR),
            ("Intersection", cv2.HISTCMP_INTERSECT),
            ("Hellinger", cv2.HISTCMP_BHATTACHARYYA)
        )
        if methodName in ("Correlation", "Intersection"):
            reverse = True
        # method = OPENCV_METHODS[methodName]
        method = cv2.HISTCMP_BHATTACHARYYA
        cmp_features = functools.partial(cv2.compareHist, method=method)
    elif feature_type == 1:
        print('using vgg16 features')
        vgg16, transform, device = create_vgg16()
        get_features = functools.partial(get_vgg16, model=vgg16, transform=transform, device=device)
        cmp_features = euclidean_dist
    elif feature_type == 2:
        print('using resnext101 features')
        resnext101, transform, device = create_resnext101()
        get_features = functools.partial(get_resnext101, model=resnext101, transform=transform, device=device)
        cmp_features = euclidean_dist
    else:
        raise AssertionError('invalid feature type: {}'.format(feature_type))

    script_filename = inspect.getframeinfo(inspect.currentframe()).filename
    script_path = os.path.dirname(os.path.abspath(script_filename))

    db = {}
    if db_file:
        db_file = os.path.join(script_path, 'log', db_file)

        if os.path.isfile(db_file):
            print('Loading feature db from {}'.format(db_file))
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
        src_file_gen = [[os.path.join(dirpath, f) for f in files]
                        for (dirpath, dirnames, files) in os.walk(_path, followlinks=True)]
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
        print('Computing features for {}/{} files ...'.format(n_new_files, n_files))
        db.update({k: (os.path.getmtime(all_stats[k]), get_features(all_stats[k]))
                   for k in tqdm(new_stats)})

    # for _stsats in new_stats:
    #     full_path = all_stats[_stsats]
    #
    #     db[_stsats] = (os.path.getmtime(full_path), curr_hist)
    #
    #     # d = cmp_features(img_features, curr_hist)
    #     # results[full_path] = d
    #
    #     # file_path_map[full_path] = full_path
    #
    #     n_files += 1
    #     sys.stdout.write('\rDone {} files'.format(n_files))
    #     sys.stdout.flush()

    print()

    all_files_features_pairs = []

    if files:
        if os.path.isfile(files):
            print('Looking for images similar to {} in {}'.format(files, paths))

            img_features = get_features(files)

            # image = cv2.imread(_filename)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # img_features = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
            #                         [0, 256, 0, 256, 0, 256])
            # cv2.normalize(img_features, img_features).flatten()
            # print('img_features: {}'.format(img_features))

            print('Comparing features...')

            results = {all_stats[k]: cmp_features(img_features, db[k][1]) for k in all_stats}
            results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
            # print('\nTotal files searched: {}'.format(n_files))

            print('Similar files found: ')
            for (i, (v, k)) in enumerate(results):
                print('{} : {} :: {}'.format(i, k, v))
                curr_image = cv2.imread(k)
                cv2.imshow('curr_image', curr_image)
                k = cv2.waitKey(0)
                if k == 27 or i >= n_results:
                    break
        elif os.path.isdir(files):
            _path = os.path.abspath(files)
            _src_file_gen = [[os.path.join(_dirpath, f) for f in _files]
                             for (_dirpath, _dirnames, _files) in os.walk(_path, followlinks=True)]
            _src_file_list = [item for sublist in _src_file_gen for item in sublist]

            if valid_exts:
                _src_file_list = [k for k in _src_file_list if
                                  os.path.splitext(os.path.basename(k).lower())[1][1:] in valid_exts]
            _all_files_list = _src_file_list

            _n_all_files = len(_all_files_list)
            print('Searching {} orig files in all'.format(_n_all_files))

            # pprint(all_files_list)

            _all_stats = {k: os.stat(k) for k in _all_files_list}
            _all_stats = {'{}_{}'.format(st.st_ino, st.st_dev): k for k, st in _all_stats.items()}

            _new_stats = [k for k in _all_stats if k not in db
                          or db[k][0] != os.path.getmtime(_all_stats[k])]

            _n_new_files = len(_new_stats)
            _n_files = len(_all_files_list)

            if _new_stats:
                print('Computing features for {}/{} orig files ...'.format(_n_new_files, _n_files))
                db.update({k: (os.path.getmtime(_all_stats[k]), get_features(_all_stats[k]))
                           for k in _new_stats})

            print('Looking for images similar to {} images in {} among {} images in {}'.format(
                _n_files, files, n_files, paths))

            print('Comparing features...')
            all_files_features_pairs = [(_all_stats[k1], all_stats[k2], cmp_features(db[k1][1], db[k2][1]))
                                        for k1 in _all_stats
                                        for k2 in all_stats
                                        if k2 not in _all_stats]
    else:
        # pairwise search
        print('Looking for pairwise similar images using thresh: {}'.format(thresh))
        print('Comparing features...')
        all_files_features_pairs = [(all_stats[k[0]], all_stats[k[1]],
                                     cmp_features(db[k[0]][1], db[k[1]][1])) for k in
                                    itertools.combinations(list(all_stats.keys()), r=2)]
    if all_files_features_pairs:
        print('Thresholding...')
        similar_img_pairs = [k for k in all_files_features_pairs if k[2] < thresh]
        print('Sorting...')
        similar_img_pairs.sort(key=lambda x: x[2])

        n_pairs = len(similar_img_pairs)
        cwd = os.getcwd()
        print('Found {} similar pairs'.format(n_pairs))
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

            print('\n1: {:50s}({}, {:5d} x {:5d})\n2: {:50s}({}, {:5d} x {:5d})\t{}'.format(
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
            elif k == ord('!'):
                print('Deleting {} and moving {} to take its place'.format(pair[0], pair[1]))
                os.remove(pair[0])
                os.rename(pair[1], pair[0])
            elif k == ord('@'):
                print('Deleting {} and moving {} to take its place'.format(pair[1], pair[0]))
                os.remove(pair[1])
                os.rename(pair[0], pair[1])
            elif k == ord('q'):
                fname, ext = os.path.splitext(os.path.basename(pair[1]))
                time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
                dst_fname = os.path.join(os.path.dirname(pair[1]), '{}_{}{}'.format(fname, time_stamp, ext))
                print('moving {} to {}'.format(pair[0], dst_fname))
                os.rename(pair[0], dst_fname)
            elif k == ord('w'):
                fname, ext = os.path.splitext(os.path.basename(pair[0]))
                time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
                dst_fname = os.path.join(os.path.dirname(pair[0]), '{}_{}{}'.format(fname, time_stamp, ext))
                print('moving {} to {}'.format(pair[1], dst_fname))
                os.rename(pair[1], dst_fname)
    # print('Files skipped: {}'.format(n_skips))
    if db_file and new_stats:
        print('Saving feature db to: {}'.format(db_file))
        pickle.dump(db, open(db_file, "wb"))


class Params:
    """
    :ivar db_file:
    :type db_file: str

    :ivar delete_file:
    :type delete_file: int

    :ivar feature_type:
        0: normalized histogram
        1: VGG16

    :type feature_type: int

    :ivar files:
    :type files: str

    :ivar root_dir:
    :type root_dir: list

    :ivar thresh:
    :type thresh: float

    """

    def __init__(self):
        self.cfg = ()
        self.db_file = ''
        self.delete_file = 0
        self.feature_type = 0
        self.files = ''
        self.root_dir = ['.']
        self.thresh = 0.1
        self.n_results = 10


def main():
    params = Params()
    paramparse.process(params)

    check_for_similar_images(params.files, params.root_dir, params.db_file,
                             params.feature_type, thresh=params.thresh, n_results=params.n_results)


if __name__ == "__main__":
    main()
