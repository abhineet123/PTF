import os
import shutil
import paramparse
import sys
from tqdm import tqdm

if __name__ == '__main__':
    params = {
        'root_dir': '.',
        'file_pattern': ['*.txt'],
        'dir_pattern': [],
        'out_name': '',
        'postfix': '',
        'switches': '-rq',
        'scp_dst': '',
        'include_all': 1,
        'relative': 1,
        'delete': 0,
        'level': 3,
    }
    paramparse.process_dict(params)
    _root_dir = params['root_dir']
    file_pattern = params['file_pattern']
    dir_pattern = params['dir_pattern']
    out_name = params['out_name']
    postfix = params['postfix']
    switches = params['switches']
    scp_dst = params['scp_dst']
    include_all = params['include_all']
    relative = params['relative']
    delete = params['delete']
    level = params['level']

    if not file_pattern:
        file_pattern = '*'

    assert os.path.isdir(_root_dir), f"invalid _root_dir: {_root_dir}"

    root_dir = os.path.abspath(_root_dir)

    if dir_pattern:
        if include_all:
            print('Restricting search to folders containing all of: {}'.format(dir_pattern))
            filter_func = lambda _sub_dirs: [x for x in _sub_dirs if all([k in x for k in dir_pattern])]
        else:
            print('Restricting search to folders containing any of: {}'.format(dir_pattern))
            filter_func = lambda _sub_dirs: [x for x in _sub_dirs if any([k in x for k in dir_pattern])]
    else:
        filter_func = lambda _sub_dirs: _sub_dirs

    print('root_dir: {}'.format(root_dir))
    sub_dirs = [root_dir,]
    # sub_dirs = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
    #             if os.path.isdir(os.path.join(root_dir, f))]
    for _ in range(level):
        sub_dirs = [os.path.join(sub_dir, f) for sub_dir in sub_dirs for f in os.listdir(sub_dir)
                    if os.path.isdir(os.path.join(sub_dir, f))]

    sub_dirs = filter_func(sub_dirs)

    all_matching_files = []
    for _dir in tqdm(sub_dirs, ncols=100):
        zip_paths = _dir
        dir_name = os.path.basename(_dir)
        parent_dir = os.path.dirname(_dir)
        out_name = '{}.zip'.format(dir_name)
        out_path = os.path.join(parent_dir, out_name)

        if relative:
            zip_cmd = f'cd {dir_name:s} && zip {switches:s} {out_path:s} *'
        else:
            zip_cmd = f'cd {root_dir:s} && zip {switches:s} {out_path:s} {dir_name:s}'
        os.system(zip_cmd)

        if delete:
            shutil.rmtree(_dir)
