import os, sys, glob, re

from Misc import processArguments, sortKey

if __name__ == '__main__':
    params = {
        'list_file': '',
        'file_name': '',
        'root_dir': '.',
        'ext': '',
        'nano_ext': '',
    }
    processArguments(sys.argv[1:], params)
    list_file = params['list_file']
    root_dir = params['root_dir']
    file_name = params['file_name']
    ext = params['ext']
    nano_ext = params['nano_ext']

    if nano_ext:
        ls_cmd = f'ls *.{nano_ext} > {list_file}'
        os.system(ls_cmd)

        nano_cmd = f'nano {list_file}'
        os.system(nano_cmd)

    if list_file:
        if os.path.isdir(list_file):
            rm_paths = [os.path.join(list_file, name) for name in os.listdir(list_file) if
                        os.path.isdir(os.path.join(list_file, name))]
            rm_paths.sort(key=sortKey)
        else:
            rm_paths = [x.strip() for x in open(list_file).readlines() if x.strip()]
            if root_dir:
                rm_paths = [os.path.join(root_dir, name) for name in rm_paths]
    else:
        rm_paths = [file_name]

    if ext:
        rm_paths = ['{}.{}'.format(name, ext) for name in rm_paths]

    for zip_path in rm_paths:
        print('removing: {}'.format(zip_path))
        rm_cmd = 'rm -rf {:s}'.format(zip_path)
        # os.system(rm_cmd)
