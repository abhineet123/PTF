import os, sys, glob, re

from Misc import processArguments, sortKey

if __name__ == '__main__':
    params = {
        'list_file': '',
        'file_name': '',
        'root_dir': '.',
    }
    processArguments(sys.argv[1:], params)
    list_file = params['list_file']
    root_dir = params['root_dir']
    file_name = params['file_name']

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

    for zip_path in rm_paths:
        print('removing: {}'.format(zip_path))
        zip_cmd = 'rm -rf {:s}'.format(zip_path)
        os.system(zip_cmd)
