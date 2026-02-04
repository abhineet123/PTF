import os
import shutil

import paramparse
from Misc import linux_path


class Params(paramparse.CFG):
    def __init__(self):
        paramparse.CFG.__init__(self, cfg_root='cfg/rustdesk')

        self.rustdesk_exe = ''
        self.connection_id = ''
        self.connection_cfg = ''
        self.rustdesk_cfg_dir = ''
        self.rustdesk_cfg_name = ''


def main():
    params: Params = paramparse.process(Params)

    if params.connection_cfg:
        assert params.rustdesk_cfg_dir, "rustdesk_cfg_dir must be provided"
        assert params.rustdesk_cfg_name, "rustdesk_cfg_name must be provided"

        assert os.path.isfile(params.connection_cfg), f"Nonexistent id_cfg: {params.connection_cfg}"

        dst_cfg_path = linux_path(params.rustdesk_cfg_dir, params.rustdesk_cfg_name)
        shutil.copy(params.connection_cfg, dst_cfg_path)

    if params.rustdesk_exe:
        assert params.connection_id, "connection_id must be provided"
        rustdesk_cmd = f'"{params.rustdesk_exe}" --connect {params.connection_id}'
        os.system(rustdesk_cmd)


if __name__ == '__main__':
    main()
