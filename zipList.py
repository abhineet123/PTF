import os
import sys
import random
from datetime import datetime
from tqdm import tqdm
import zipfile
import piexif

import paramparse


def title_from_exif(img_path):
    exif_dict = piexif.load(img_path)
    title = exif_dict['0th'][piexif.ImageIFD.ImageDescription].decode("utf-8")

    return title

def to_epoch(dst_fname_noext, fmt='%Y_%m_%d-%H_%M_%S-%f', is_path=True):
    if is_path:
        dst_fname_noext = os.path.splitext(os.path.basename(dst_fname_noext))[0]

    timestamp_ms, _id = dst_fname_noext.split('__')
    timestamp_str = timestamp_ms + '000'
    timestamp = datetime.strptime(timestamp_str, fmt)
    epoch_sec = datetime.timestamp(timestamp)
    epoch = str(int(float(epoch_sec) * 1000))

    return epoch, _id


def from_epoch(epoch, src_id, fmt='%Y_%m_%d-%H_%M_%S-%f'):
    epoch_sec = float(epoch) / 1000.0
    timstamp = datetime.fromtimestamp(epoch_sec)
    timstamp_str = timstamp.strftime(fmt)
    timstamp_str_ms = timstamp_str[:-3]
    timstamp_id = '{}__{:d}'.format(timstamp_str_ms, src_id)

    return timstamp_id


if __name__ == '__main__':
    params = dict(
        list_file='',
        exclude_list='',
        root_dir='',
        out_name='',
        scp_dst='',
        postfix='',
        relative=0,
        n_samples=0,
        shuffle=0,
        scp_port='',
        inclusion='',
        move_to_home=1,
        add_time_stamp=1,
        recursive=1,
        builtin=1,
        name_from_title=0,
        switches='-r',
    )
    paramparse.process_dict(params)

    list_file = params['list_file']
    exclude_list = params['exclude_list']
    root_dir = params['root_dir']
    out_name = params['out_name']
    postfix = params['postfix']
    scp_dst = params['scp_dst']
    relative = params['relative']
    switches = params['switches']
    shuffle = params['shuffle']
    n_samples = params['n_samples']
    scp_port = params['scp_port']
    move_to_home = params['move_to_home']
    add_time_stamp = params['add_time_stamp']
    builtin = params['builtin']
    recursive = params['recursive']
    name_from_title = params['name_from_title']

    excluded_names = []

    if os.path.isdir(list_file):
        print(f'looking for zip paths in {list_file}')

        if recursive:
            print(f'searching recursively')
            zip_paths_gen = [[os.path.join(dirpath, f) for f in filenames]
                             for (dirpath, dirnames, filenames) in os.walk(list_file, followlinks=False)]
            zip_paths = [item for sublist in zip_paths_gen for item in sublist]
        else:
            zip_paths = [os.path.join(list_file, f) for f in os.listdir(list_file)]

        zip_paths.sort()

    elif os.path.isfile(list_file):
        print(f'reading zip paths from {list_file}')
        zip_paths = [x.strip() for x in open(list_file).readlines() if x.strip()
                     and not x.startswith('#')
                     and not x.startswith('@')
                     and x.strip() not in excluded_names
                     ]
        if root_dir:
            zip_paths = [os.path.join(root_dir, name) for name in zip_paths]

    else:
        raise AssertionError(f'invalid list file: {list_file}')

    if name_from_title:
        zip_paths = [f for f in zip_paths if f.endswith('.jpg')]

    n_paths = len(zip_paths)
    print(f'found {n_paths} zip paths')

    if exclude_list:
        if os.path.isdir(exclude_list):
            print(f'looking for excluded file names in {exclude_list}')

            excluded_paths = [os.path.join(exclude_list, f) for f in os.listdir(exclude_list)]

            excluded_paths.sort()

        elif os.path.isfile(exclude_list):
            print(f'reading excluded file names from {exclude_list}')
            excluded_paths = [x.strip() for x in open(exclude_list).readlines() if x.strip()
                              and not x.startswith('#')
                              and not x.startswith('@')
                              ]
        else:
            raise AssertionError(f'invalid exclude_list: {exclude_list}')

        if name_from_title:
            print('getting names from titles')

            excluded_paths = [f for f in excluded_paths if f.endswith('.jpg')]
            # excluded_names = [title_from_exif(k) for k in excluded_paths]
            # zip_names = [title_from_exif(k) for k in zip_paths]
            excluded_names = [to_epoch(k) for k in excluded_paths]
            zip_names = [to_epoch(k) for k in zip_paths]

        else:
            excluded_names = [os.path.basename(k) for k in excluded_paths]
            zip_names = [os.path.basename(k) for k in zip_paths]

        n_excluded_files = len(excluded_names)
        print()
        print(f'found {n_excluded_files} excluded_files')

        useless_excluded_paths = [k for k, n in zip(excluded_paths, excluded_names) if n not in zip_names]

        with open('useless_excluded_paths.txt', 'w') as fid:
            fid.write('\n'.join(useless_excluded_paths))

        n_useless_excluded_paths = len(useless_excluded_paths)

        zip_paths = [k for k, n in zip(zip_paths, zip_names) if n not in excluded_names]

        _n_paths = len(zip_paths)
        n_filtered = n_paths - _n_paths
        n_unfiltered = n_excluded_files - n_filtered
        print(f'found {_n_paths} zip paths after filtering')
        print(f'filtered: {n_filtered}')
        print(f'unfiltered: {n_unfiltered}')
        print(f'useless_excluded_paths: {n_useless_excluded_paths}')
        print()

        n_paths = _n_paths

    if not out_name:
        dir_names = list_file.split(os.sep)

        for _dir in dir_names:
            out_name = '{}_{}'.format(out_name, _dir) if out_name else _dir

        out_name = out_name.replace('.', '_')
        out_name = out_name.replace('(', '_')
        out_name = out_name.replace(')', '_')

    if n_paths > n_samples > 0:
        if shuffle:
            print('shuffling zip paths')
            random.shuffle(zip_paths)

            out_name = '{}_shuffle'.format(out_name)

        print(f'Sampling {n_samples} / {n_paths} zip paths')

        out_name = '{}_sample_{}'.format(out_name, n_samples)

        zip_paths = zip_paths[:n_samples]

    if not root_dir:
        root_dir = os.path.abspath(os.path.dirname(zip_paths[0]))

    # if not out_name:
    #     _root_dir = os.path.basename(os.path.dirname(os.path.abspath(zip_paths[0])))
    #     list_fname_no_ext = os.path.splitext(os.path.basename(list_file))[0]
    #     out_name = '{}_{}'.format(_root_dir, list_fname_no_ext)
    # else:
    #     out_name = os.path.splitext(out_name)[0]
    #
    # if postfix:
    #     out_name = '{}_{}'.format(out_name, postfix)
    #
    # time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    # out_name = '{}_{}.zip'.format(out_name, time_stamp)

    if postfix:
        out_name = '{}_{}'.format(out_name, postfix)

    if add_time_stamp:
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        out_name = '{}_{}'.format(out_name, time_stamp)

    out_name = '{}.zip'.format(out_name)

    if relative:
        out_path = os.path.join(root_dir, out_name)
    else:
        out_path = out_name

    n_paths = len(zip_paths)

    if builtin:
        print('writing {} files to {}'.format(n_paths, out_path))
        with zipfile.ZipFile(out_path, mode="w") as archive:
            for zip_path in tqdm(zip_paths):
                if relative:
                    archive.write(zip_path, zip_path)
                else:
                    archive.write(zip_path, os.path.basename(zip_path))
    else:
        zip_paths = ['"{}"'.format(k) for k in zip_paths]

        zip_cmd = ' '.join(zip_paths)
        # for zip_path in zip_paths:
        #     zip_cmd = '{:s} {:s}'.format(zip_cmd, zip_path)

        if relative:
            zip_cmd = 'cd {} && zip {} {} . -i {}'.format(root_dir, switches, out_name, zip_cmd)
        else:
            zip_cmd = 'zip {:s} {:s} {:s}'.format(switches, out_name, zip_cmd)

        # print('\nrunning: {}\n'.format(zip_cmd))

        os.system(zip_cmd)

    assert os.path.exists(out_path), "zipping failed: {}".format(out_path)

    # os.system('unzip -l {}'.format(out_path))

    if scp_dst:
        scp_cmd = 'scp'
        if scp_port:
            scp_cmd = '{} -P {}'.format(scp_cmd, scp_port)

        scp_cmd = '{} "{}" {}:~/'.format(scp_cmd, out_path, scp_dst)

        print('\nrunning: {}\n'.format(scp_cmd))
        os.system(scp_cmd)
        rm_cmd = 'rm "{}"'.format(out_path)
        print('\nrunning: {}\n'.format(rm_cmd))
        os.system(rm_cmd)
    elif move_to_home:
        mv_cmd = 'mv "{:s}" ~'.format(out_path)
        print('\nrunning: {}\n'.format(mv_cmd))
        os.system(mv_cmd)

    print('out_name:\n {}'.format(out_name))
