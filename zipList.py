import os
import sys
import random
from datetime import datetime

from Misc import processArguments, sortKey

if __name__ == '__main__':
    params = {
        'list_file': '',
        'root_dir': '',
        'out_name': '',
        'scp_dst': '',
        'postfix': '',
        'relative': 0,
        'n_samples': 0,
        'shuffle': 0,
        'scp_port': '',
        'inclusion': '',
        'add_time_stamp': 1,
        'move_to_home': 1,
        'add_time_stamp': 1,
        'switches': '-r',
    }
    processArguments(sys.argv[1:], params)
    list_file = params['list_file']
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

    if os.path.isdir(list_file):
        print(f'looking for zip paths in {list_file}')

        zip_paths = [os.path.join(list_file, name) for name in os.listdir(list_file)]
        zip_paths.sort(key=sortKey)

    elif os.path.isfile(list_file):
        print(f'reading zip paths from {list_file}')
        zip_paths = [x.strip() for x in open(list_file).readlines() if x.strip()
                     and not x.startswith('#')
                     and not x.startswith('@')
                     ]
        if root_dir:
            zip_paths = [os.path.join(root_dir, name) for name in zip_paths]

    else:
        raise AssertionError('invalid list file: {}')

    n_paths = len(zip_paths)
    print(f'found {n_paths} zip paths')

    if not out_name:
        dir_names = list_file.split(os.sep)

        for _dir in dir_names:
            out_name = '{}_{}'.format(out_name, _dir) if out_name else _dir

        if postfix:
            out_name = '{}_{}'.format(out_name, postfix)

        out_name = out_name.replace('.', '_')
        out_name = out_name.replace('(', '_')
        out_name = out_name.replace(')', '_')

    if n_paths > n_samples > 0:
        print(f'Sampling {n_samples} / {n_paths} zip paths')

        out_name = '{}_sample_{}'.format(out_name, n_samples)

        if shuffle:
            print('shuffling zip paths')
            random.shuffle(zip_paths)

            out_name = '{}_shuffle'.format(out_name)

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

    if add_time_stamp:
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        out_name = '{}_{}'.format(out_name, time_stamp)

    out_name = '{}.zip'.format(out_name)

    zip_paths = ['"{}"'.format(k) for k in zip_paths]

    zip_cmd = ' '.join(zip_paths)
    # for zip_path in zip_paths:
    #     zip_cmd = '{:s} {:s}'.format(zip_cmd, zip_path)

    if relative:
        zip_cmd = 'cd {} && zip {} {} . -i {}'.format(root_dir, switches, out_name, zip_cmd)
        out_path = os.path.join(root_dir, out_name)
    else:
        zip_cmd = 'zip {:s} {:s} {:s}'.format(switches, out_name, zip_cmd)
        out_path = out_name

    print('\nrunning: {}\n'.format(zip_cmd))

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
