import os, sys
from datetime import datetime

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

    # if src_fname.startswith('.'):

    if src_fname.endswith(os.sep):
        src_dir = src_fname
    else:
        src_dir = os.path.dirname(src_fname)

    if not os.path.isdir(src_dir):
        print('Creating folder: {}'.format(src_dir))
        os.makedirs(src_dir)

    scp_cmd = 'scp -r {}:{} {}'.format(scp_dst, src_fname, src_fname)
    print('\nrunning: {}\n'.format(scp_cmd))
    os.system(scp_cmd)


