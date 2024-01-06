import os, sys, glob, re

from Misc import processArguments, sortKey

if __name__ == '__main__':
    params = {
        'list_file': '',
        'file_name': '',
        'root_dir': '.',
        'scp_dst': '',
    }
    processArguments(sys.argv[1:], params)
    list_file = params['list_file']
    root_dir = params['root_dir']
    file_name = params['file_name']
    scp_dst = params['scp_dst']

    if list_file:
        if os.path.isdir(list_file):
            scp_paths = [os.path.join(list_file, name) for name in os.listdir(list_file) if
                         os.path.isdir(os.path.join(list_file, name))]
            scp_paths.sort(key=sortKey)
        else:
            scp_paths = [x.strip() for x in open(list_file).readlines() if x.strip()]
            if root_dir:
                root_dir = os.path.abspath(root_dir)
                scp_paths = [os.path.join(root_dir, name) for name in scp_paths]
    else:
        scp_paths = [file_name]

    for scp_path in scp_paths:
        scp_cmd = 'scp -r {}:{} {}'.format(scp_dst, scp_path, scp_path)
        print('\nrunning: {}\n'.format(scp_cmd))
        os.system(scp_cmd)
