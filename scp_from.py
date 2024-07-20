import os

import paramparse

from Misc import linux_path


class Params:

    def __init__(self):
        self.cfg = ()
        self.abs_path = 1
        self.copy_links = 0
        self.file_mode = 0
        self.src_list = ''
        self.overwrite = 0
        self.scp_dst = ''
        self.scp_port = ''
        self.src_ext = ''
        self.src_fname = ''
        self.inverse = 0
        self.remove_src = 0
        self.verbose = 1


def main():
    params = Params()
    paramparse.process(params)

    if params.src_list:
        from tqdm import tqdm
        print(f'reading sources from {params.src_list}')
        srcs = open(params.src_list, 'r').readlines()
        params.verbose=0
        pbar = tqdm(srcs, total=len(srcs))
        for src in pbar:
            src = src.strip()
            pbar.set_description(src)
            params.src_fname = src
            try:
                run(params)
            except KeyboardInterrupt:
                break
    else:
        run(params)


def run(params: Params):
    src_fname = params.src_fname
    scp_dst = params.scp_dst
    scp_port = params.scp_port
    overwrite = params.overwrite
    file_mode = params.file_mode
    abs_path = params.abs_path
    copy_links = params.copy_links
    remove_src = params.remove_src
    src_ext = params.src_ext

    import platform

    is_wsl = platform.uname().release.endswith("microsoft-standard-WSL2")

    # src_fname = os.path.realpath(src_fname)
    # src_fname_abs = os.popen(f'realpath -s {src_fname}').read()
    # src_fname_abs = os.path.abspath(src_fname)
    src_dir_abs = str(os.popen('pwd').read().strip())

    if params.verbose:
        print(f'src_dir_abs: {src_dir_abs}')

    if is_wsl and src_dir_abs.startswith('/mnt/'):
        _src_dir = src_dir_abs
        while True:
            git_dir = linux_path(_src_dir, '.git')
            is_dir = os.path.isdir(git_dir)
            if is_dir:
                break
            _src_dir = os.path.dirname(_src_dir)

            if _src_dir == os.path.dirname(_src_dir):
                raise AssertionError('reached filesystem root without finding a git folder')

        dir_to_replace = os.path.dirname(_src_dir)
        src_dir_abs = src_dir_abs.replace(dir_to_replace, os.path.expanduser('~'))
        if not os.path.isdir(src_dir_abs):
            src_dir_abs = src_dir_abs.lower()

        assert os.path.isdir(src_dir_abs), f"nonexistent dir: {src_dir_abs}"

    src_fname_abs = linux_path(src_dir_abs, src_fname)

    src_fname_no_ext, src_fname_ext = os.path.splitext(os.path.basename(src_fname))

    home_path = os.path.abspath(os.path.expanduser("~"))
    if src_fname_abs.startswith(home_path):
        # src_fname_rel = os.path.relpath(src_fname, home_path)
        # scp_fname = linux_path('~', src_fname_rel)
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

    if params.verbose:
        print('src_fname_abs: {}'.format(src_fname_abs))
        print('src_fname: {}'.format(src_fname))
        print('home_path: {}'.format(home_path))
        print('src_fname_rel: {}'.format(src_fname_rel))
        print('scp_fname: {}'.format(scp_fname))

    src_dir = os.path.dirname(src_fname_abs)
    if src_dir and not os.path.isdir(src_dir):
        if params.verbose:
            print('Creating folder: {}'.format(src_dir))
        os.makedirs(src_dir, exist_ok=True)

    switches = '-r --mkpath'

    if params.verbose:
        switches = f'{switches} -v --progress'

    if src_ext:
        switches = f' --include="*/" --include="*.{src_ext}" --exclude="*" {switches}'

    # switches += ' --dry-run'

    if copy_links:
        switches += ' --copy-links'
    else:
        switches += ' --no-links'

    if not overwrite:
        switches += ' --ignore-existing'

    if remove_src:
        switches += ' --remove-source-files'

    # if overwrite:
    #     rsync_cmd = 'scp -r {}:{} {}'.format(scp_dst, scp_fname, src_fname)

    if '*' in src_fname:
        dst_fname = './'
    else:
        dst_fname = src_fname

    if params.inverse:
        rsync_cmd = f'rsync {switches} {dst_fname} {scp_dst}:{scp_fname}'
    else:
        rsync_cmd = f'rsync {switches} {scp_dst}:{scp_fname} {dst_fname}'

    if scp_port:
        rsync_cmd = f"{rsync_cmd} -e 'ssh -p {scp_port}'"

    if params.verbose:
        print(f'\nrunning: {rsync_cmd}\n')

    try:
        os.system(rsync_cmd)
    except KeyboardInterrupt as e:
        raise e


if __name__ == '__main__':
    main()
