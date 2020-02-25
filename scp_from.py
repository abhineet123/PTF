import os, sys
from os.path import expanduser

from Misc import processArguments

if __name__ == '__main__':
    params = {
        'src_fname': '',
        'scp_dst': '',
    }
    processArguments(sys.argv[1:], params)
    src_fname = params['src_fname']
    scp_dst = params['scp_dst']

    src_fname = os.path.abspath(src_fname)
    src_dir = os.path.dirname(src_fname)

    home = expanduser("~")
    src_fname_rel = os.path.relpath(src_fname, home)

    print('home: {}'.format(home))
    print('src_fname_rel: {}'.format(src_fname_rel))

    scp_fname = os.path.join('~', src_fname_rel)

    if not os.path.isdir(src_dir):
        print('Creating folder: {}'.format(src_dir))
        os.makedirs(src_dir)

    scp_cmd = 'scp -r {}:{} {}'.format(scp_dst, scp_fname, src_fname)
    print('\nrunning: {}\n'.format(scp_cmd))
    os.system(scp_cmd)


