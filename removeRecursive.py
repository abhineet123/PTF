import os, sys
import glob

from Misc import processArguments

if __name__ == '__main__':
    params = {
        'root_dir': '.',
        'file_pattern': '*.txt',
        'postfix': '',
        'switches': '-rf',
    }
    processArguments(sys.argv[1:], params)
    root_dir = params['root_dir']
    file_pattern = params['file_pattern']
    postfix = params['postfix']
    switches = params['switches']

    root_dir = os.path.abspath(root_dir)

    print('root_dir: ', root_dir)
    print('file_pattern: ', file_pattern)

    sub_dirs_gen = [[os.path.join(dirpath, f) for f in dirnames]
                    for (dirpath, dirnames, filenames) in os.walk(root_dir, followlinks=True)]
    sub_dirs = [os.path.relpath(item, root_dir) for sublist in sub_dirs_gen for item in sublist]

    # print('sub_dirs:\n{}'.format(sub_dirs))

    # sub_dirs = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]

    sub_dirs.append('.')

    # rm_paths = ''
    for _dir in sub_dirs:
        current_path = os.path.join(_dir, file_pattern) if file_pattern else _dir

        full_path = os.path.join(root_dir, current_path)

        matches = glob.glob(full_path)
        for _match in matches:
            rm_cmd = 'rm {:s} {:s}'.format(switches, _match)
            print('running: {}\n'.format(rm_cmd))
            # os.system(rm_cmd)

        # rm_cmd = 'cd {:s} && rm {:s} {:s}'.format(root_dir, switches, current_path)
        # print('running: {}\n'.format(rm_cmd))
        # os.system(rm_cmd)

        # rm_paths = '{} {}'.format(rm_paths, current_path) if rm_paths else current_path

    # rm_paths.replace(root_dir, '')
    # print('rm_paths: ', rm_paths)

    # rm_cmd = 'cd {:s} && rm {:s} {:s}'.format(root_dir, switches, rm_paths)
    # print('\nrunning: {}\n'.format(rm_cmd))
    # os.system(rm_cmd)

