import os
import subprocess
import paramparse
import stat
from datetime import datetime

from Misc import linux_path


class Params(paramparse.CFG):

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix="snapshot_to_html")
        self.exe_path = ''
        self.src_path = ''
        self.dst_path = ''
        self.out_name = ''
        self.excluded_names = []
        self.exclude_links = 1
        self.exclude_hidden = 1
        self.extra_info = 1
        self.verbose = 1


def is_link(src):
    child = subprocess.Popen(
        "fsutil reparsepoint query \"{}\"".format(src),
        stdout=subprocess.PIPE
    )
    streamdata = child.communicate()[0]
    rc = child.returncode

    if rc == 0:
        return True
    return False


def is_hidden(filepath):
    """
    https://stackoverflow.com/a/6365265
    """
    name = os.path.basename(os.path.abspath(filepath))
    return name.startswith('.') or has_hidden_attribute(filepath)


def has_hidden_attribute(filepath):
    return bool(os.stat(filepath).st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)


def main():
    # input("press any key")

    params: Params = paramparse.process(Params)

    assert params.exe_path, "exe_path must be provided"
    assert params.src_path, "src_path must be provided"
    assert params.out_name, "out_name must be provided"

    src_dirs = [k for k in os.listdir(params.src_path)
                if k not in params.excluded_names]

    if not params.dst_path:
        params.dst_path = params.src_path

    n_src_dirs = len(src_dirs)

    if not src_dirs:
        print(f"\nno valid src_dirs found in: {params.src_path}\n")
        return

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir_path = linux_path(params.dst_path, f"{params.out_name}_{timestamp}")
    os.makedirs(out_dir_path, exist_ok=True)

    for src_dir_id, src_dir in enumerate(src_dirs):
        src_dir_path = linux_path(params.src_path, src_dir)
        if not os.path.isdir(src_dir_path):
            print(f"\n{src_dir_id + 1}/{n_src_dirs} excluding file: {src_dir}\n")
            continue

        if params.exclude_hidden and is_hidden(src_dir_path):
            print(f"\n{src_dir_id + 1}/{n_src_dirs} excluding hidden: {src_dir}\n")
            continue

        if params.exclude_links and is_link(src_dir_path):
            print(f"\n{src_dir_id + 1}/{n_src_dirs} excluding link: {src_dir}\n")
            continue

        out_name = f"{params.out_name}-{src_dir}"
        out_path = linux_path(out_dir_path, f"{out_name}.html")
        cmd = f'{params.exe_path} -path:"{src_dir_path}" -outfile:"{out_path}" -title:"{out_name}" --silent'

        print(f"{src_dir_id + 1}/{n_src_dirs} {src_dir}: {cmd}")
        os.system(cmd)
        input("press any key")

    # input("press any key")


if __name__ == '__main__':
    main()
