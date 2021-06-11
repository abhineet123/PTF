import os, sys
from os.path import expanduser

from Misc import processArguments

if __name__ == '__main__':
    params = {
        'src_fname': '',
        'scp_dst': '',
        'overwrite': 0,
        'file_mode': 0,
    }
    processArguments(sys.argv[1:], params)
    src_fname = params['src_fname']
    scp_dst = params['scp_dst']
    overwrite = params['overwrite']
    file_mode = params['file_mode']

    # src_fname = os.path.realpath(src_fname)
    src_fname_abs = os.path.abspath(src_fname)

    src_fname_no_ext, src_fname_ext = os.path.splitext(os.path.basename(src_fname))

    home_path = os.path.abspath(expanduser("~"))
    if src_fname_abs.startswith(home_path):
        src_fname_rel = os.path.relpath(src_fname, home_path)
        scp_fname = os.path.join('~', src_fname_rel)
    else:
        src_fname_rel = src_fname
        scp_fname = src_fname_abs

    if not file_mode and not src_fname_ext and not src_fname.endswith('/'):
        """no ext --> directory -> add terminating / to prevent recreation of directory structure on dst"""
        src_fname += '/'
        scp_fname += '/'

    print('src_fname_abs: {}'.format(src_fname_abs))
    print('src_fname: {}'.format(src_fname))
    print('home_path: {}'.format(home_path))
    print('src_fname_rel: {}'.format(src_fname_rel))
    print('scp_fname: {}'.format(scp_fname))

    src_dir = os.path.dirname(src_fname_abs)
    if src_dir and not os.path.isdir(src_dir):
        print('Creating folder: {}'.format(src_dir))
        os.makedirs(src_dir, exist_ok=1)

    switches = '-r -v --progress'
    if not overwrite:
        switches += ' --ignore-existing'

    # if overwrite:
    #     rsync_cmd = 'scp -r {}:{} {}'.format(scp_dst, scp_fname, src_fname)

    rsync_cmd = 'rsync {} {}:{} {}'.format(switches, scp_dst, scp_fname, src_fname)

    print('\nrunning: {}\n'.format(rsync_cmd))
    os.system(rsync_cmd)


