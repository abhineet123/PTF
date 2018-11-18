import os, sys
from datetime import datetime
from pprint import pprint

from Misc import processArguments

if __name__ == '__main__':
    params = {
        'root_dir': '.',
        'file_pattern': '*.txt',
        'dir_pattern': [],
        'out_name': '',
        'postfix': '',
        'switches': '-rq',
        'scp_dst': '',
    }
    processArguments(sys.argv[1:], params)
    _root_dir = params['root_dir']
    file_pattern = params['file_pattern']
    dir_pattern = params['dir_pattern']
    out_name = params['out_name']
    postfix = params['postfix']
    switches = params['switches']
    scp_dst = params['scp_dst']

    if os.path.isdir(_root_dir):
        root_dirs = [_root_dir]
        root_base_dir = _root_dir
    else:
        root_base_dir = os.path.dirname(_root_dir)
        _dir_pattern = os.path.basename(_root_dir)
        root_dirs = [os.path.join(root_base_dir, f) for f in os.listdir(root_base_dir) if _dir_pattern in f and
                     os.path.isdir(os.path.join(root_base_dir, f))]
        if not root_dirs:
            raise IOError('No root directories found matching {} in {}'.format(_dir_pattern, root_base_dir))

    print('file_pattern: {}'.format(file_pattern))
    print('root_base_dir: {}'.format(root_base_dir))

    if not out_name:
        dir_names = root_base_dir.split(os.sep)
        for _dir in dir_names:
            out_name = '{}_{}'.format(out_name, _dir) if out_name else _dir

    root_base_dir = os.path.abspath(root_base_dir)
    sub_dirs = []
    for root_dir in root_dirs:
        root_dir = os.path.abspath(root_dir)
        print('root_dir: {}'.format(root_dir))
        sub_dirs_gen = [[os.path.join(dirpath, f) for f in dirnames]
                        for (dirpath, dirnames, filenames) in os.walk(root_dir, followlinks=True)]
        sub_dirs += [os.path.relpath(item, root_base_dir) for sublist in sub_dirs_gen for item in sublist]

    if len(dir_pattern) == 1 and not dir_pattern[0]:
        dir_pattern = []
        
    if dir_pattern:
        print('Restricting search to folders containing:{}'.format(dir_pattern))
        check = lambda x: all([k in x for k in dir_pattern])
        sub_dirs = [x for x in sub_dirs if check(x)]

    print('sub_dirs:\n')
    pprint(sub_dirs)

    sub_dirs.append('.')
    zip_paths = ''
    for _dir in sub_dirs:
        current_path = os.path.join(_dir, file_pattern) if file_pattern else _dir
        zip_paths = '{} {}'.format(zip_paths, current_path) if zip_paths else current_path

    # zip_paths.replace(root_dir, '')

    # print('zip_paths:\n')
    # pprint(zip_paths)



    if postfix:
        out_name = '{}_{}'.format(out_name, postfix)

    out_name.replace('.', '_')
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_name = '{}_{}.zip'.format(out_name, time_stamp)

    zip_cmd = 'cd {:s} && zip {:s} {:s} {:s}'.format(root_base_dir, switches, out_name, zip_paths)

    # print('\nrunning: {}\n'.format(zip_cmd))
    # subprocess.call(zip_cmd)
    os.system(zip_cmd)

    if scp_dst:
        scp_cmd = 'cd {:s} && scp {} {}:~/'.format(root_base_dir, out_name, scp_dst)
        print('\nrunning: {}\n'.format(scp_cmd))
        os.system(scp_cmd)
        rm_cmd = 'cd {:s} && rm {}'.format(root_base_dir, out_name)
        print('\nrunning: {}\n'.format(rm_cmd))
        os.system(rm_cmd)
    else:
        mv_cmd = 'cd {:s} && mv {:s} ~'.format(root_base_dir, out_name)
        print('\nrunning: {}\n'.format(mv_cmd))
        os.system(mv_cmd)

    print('out_name:\n {}'.format(out_name))
