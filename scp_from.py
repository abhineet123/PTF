import os, sys
from os.path import expanduser

from Misc import processArguments

if __name__ == '__main__':
    params = {
        'src_fname': '',
        'scp_dst': '',
        'overwrite': 0,
    }
    processArguments(sys.argv[1:], params)
    src_fname = params['src_fname']
    scp_dst = params['scp_dst']
    overwrite = params['overwrite']

    src_fname_no_ext, src_fname_ext = os.path.splitext(os.path.basename(src_fname))

    if not src_fname_ext and not src_fname.endswith('/'):
        """no ext --> directory -> add terminating / to prevent recreation of directory structure on dst"""
        src_fname += '/'

    src_fname = os.path.realpath(src_fname)
    src_fname_abs = os.path.abspath(src_fname)
    src_dir = os.path.dirname(src_fname)

    home = os.path.abspath(expanduser("~"))
    src_fname_rel = os.path.relpath(src_fname, home)

    print('src_fname_abs: {}'.format(src_fname_abs))
    print('src_fname: {}'.format(src_fname))
    print('home: {}'.format(home))
    print('src_fname_rel: {}'.format(src_fname_rel))

    scp_fname = os.path.join('~', src_fname_rel)

    if not os.path.isdir(src_dir):
        print('Creating folder: {}'.format(src_dir))
        os.makedirs(src_dir)

    if overwrite:
        scp_cmd = 'scp -r {}:{} {}'.format(scp_dst, scp_fname, src_fname)
    else:
        scp_cmd = 'rsync -r --ignore-existing {}:{} {}'.format(scp_dst, scp_fname, src_fname)

    print('\nrunning: {}\n'.format(scp_cmd))
    os.system(scp_cmd)


