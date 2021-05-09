import time
import paramparse
import os, shutil
from datetime import datetime
import paramiko
import subprocess

from Misc import linux_path, add_suffix


class Params:
    class SCP:
        auth_path = ""
        url = ""
        user = ""
        pwd = ""
        home_dir = "/home/abhineet"
        code_path = 'deep_mdp/tracking_module'

        show_output = 1
        enable_zipping = 1
        remove_src = 1
        rename_src = 1

        def read_auth(self):
            if not self.auth_path:
                print("auth_path is not provided")
                return

            auth_data = open(self.auth_path, 'r').readlines()
            auth_data = [k.strip() for k in auth_data]
            self.url, self.user, self.pwd = auth_data[0].split(' ')

    working_dir = ''
    log_dir = 'log'
    log_fname = 'mot_metrics_accumulative_hota.log'

    servers = ['grs', 'x99', 'orca']
    # servers = ['x99', 'orca']
    cmd_in_file = linux_path('log', 'multi_vis_cmd.txt')
    force_download = 0
    remove_header = 1
    scp = SCP()


def run_unzip(tee_zip_path):
    unzip_cmd = 'unzip {}'.format(tee_zip_path)
    unzip_cmd_list = unzip_cmd.split(' ')
    print('Running {}'.format(unzip_cmd))
    subprocess.check_call(unzip_cmd_list)


def run_scp(params):
    """

    :param Params.SCP params:
    :param server_name:
    :param dst_rel_path:
    :param is_file:
    :return:
    """

    assert params.pwd and params.user and params.url, "auth data not provided"

    src_root_path = '/var/mobile/Media/DCIM/100APPLE'

    remote_cmd = "cd {} && ls".format(src_root_path)

    if params.show_output == 2:
        print('running {}'.format(remote_cmd))

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(params.url, username=params.user, password=params.pwd)

    stdin, stdout, stderr = client.exec_command(remote_cmd)
    stdout = list(stdout)
    stderr = list(stderr)

    if params.show_output:
        for line in stdout:
            print(line.strip('\n'))

    if stderr:
        # for line in stderr:
        #     print(line.strip('\n'))
        raise AssertionError('remote command did not work' + '\n' + '\n'.join(stderr))

    for _file in stdout:
        _file_path = linux_path(src_root_path, _file.strip())
        scp_cmd = "pscp -pw {} -r -P 22 {}@{}:{} ./".format(params.pwd, params.user, params.url, _file_path)
        if params.show_output == 2:
            print('Running {}'.format(scp_cmd))

        scp_cmd_list = scp_cmd.split(' ')
        subprocess.check_call(scp_cmd_list)

        if params.remove_src:

            remote_cmd = "rm {}".format(_file_path)

            if params.show_output == 2:
                print('running {}'.format(remote_cmd))

            stdin, stdout, stderr = client.exec_command(remote_cmd)
            stdout = list(stdout)
            stderr = list(stderr)

            if params.show_output:
                for line in stdout:
                    print(line.strip('\n'))

            if stderr:
                raise AssertionError('remote command did not work' + '\n' + '\n'.join(stderr))
    else:
        print('no images found')

    client.close()


def main():
    params = Params()

    paramparse.process(params, allow_unknown=1)

    params.scp.read_auth()

    if params.working_dir:
        os.chdir(params.working_dir)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")

    while True:
        print('\n' + timestamp + '\n')
        run_scp(params.scp)


if __name__ == '__main__':
    main()
