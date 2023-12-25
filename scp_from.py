import os

import paramparse


class Params:

    def __init__(self):
        self.cfg = ()
        self.abs_path = 1
        self.copy_links = 0
        self.file_mode = 0
        self.overwrite = 0
        self.scp_dst = ''
        self.scp_port = ''
        self.src_fname = ''


def main():
    params = Params()
    paramparse.process(params)

    src_fname = params.src_fname
    scp_dst = params.scp_dst
    scp_port = params.scp_port
    overwrite = params.overwrite
    file_mode = params.file_mode
    abs_path = params.abs_path
    copy_links = params.copy_links

    # src_fname = os.path.realpath(src_fname)
    # src_fname_abs = os.popen(f'realpath -s {src_fname}').read()
    # src_fname_abs = os.path.abspath(src_fname)
    src_dir_abs = str(os.popen('pwd').read().strip())
    src_fname_abs = os.path.join(src_dir_abs, src_fname)

    src_fname_no_ext, src_fname_ext = os.path.splitext(os.path.basename(src_fname))

    home_path = os.path.abspath(os.path.expanduser("~"))
    if src_fname_abs.startswith(home_path):
        # src_fname_rel = os.path.relpath(src_fname, home_path)
        # scp_fname = os.path.join('~', src_fname_rel)
        src_fname_rel = scp_fname = src_fname_abs.replace(home_path, '~')
    else:
        src_fname_rel = src_fname
        if abs_path:
            scp_fname = src_fname_abs
        else:
            scp_fname = src_fname

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
        os.makedirs(src_dir, exist_ok=True)

    switches = '-r -v --progress'
    if copy_links:
        switches += ' --copy-links'
    else:
        switches += ' --no-links'

    if not overwrite:
        switches += ' --ignore-existing'

    # if overwrite:
    #     rsync_cmd = 'scp -r {}:{} {}'.format(scp_dst, scp_fname, src_fname)

    if '*' in src_fname:
        dst_fname = './'
    else:
        dst_fname = src_fname
    rsync_cmd = 'rsync {} {}:{} {}'.format(switches, scp_dst, scp_fname, dst_fname)

    if scp_port:
        rsync_cmd = "{} -e 'ssh -p {}'".format(rsync_cmd, scp_port)

    print('\nrunning: {}\n'.format(rsync_cmd))
    os.system(rsync_cmd)


if __name__ == '__main__':
    main()
