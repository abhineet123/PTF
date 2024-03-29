import platform
import os

from Misc import linux_path

is_wsl = platform.uname().release.endswith("microsoft-standard-WSL2")

# src_fname = os.path.realpath(src_fname)
# src_fname_abs = os.popen(f'realpath -s {src_fname}').read()
# src_fname_abs = os.path.abspath(src_fname)
src_dir_abs = str(os.popen('pwd').read().strip())

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
    print(f'src_dir_abs: {src_dir_abs}')

    os.system(f'tmux send-keys "cd {src_dir_abs}" Enter')

