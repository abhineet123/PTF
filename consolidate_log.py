import os
import sys
import subprocess
import numpy as np
import pandas as pd

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk

from zipfile import ZipFile
from datetime import datetime

import logging
import shutil
import paramiko
import cv2

# import difflib

logging.getLogger("paramiko").setLevel(logging.WARNING)

import paramparse

from pprint import pformat

from Misc import linux_path, add_suffix


class Params:
    class SCP:
        auth_path = ""
        url = ""
        user = ""
        pwd = ""
        home_dir = "/home/abhineet"
        code_path = 'deep_mdp/tracking_module'

        show_output = 0
        enable_zipping = 1
        remove_zip = 1
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

    # servers = ['grs', 'x99', 'orca']
    servers = ['x99', 'orca']
    cmd_in_file = linux_path('log', 'multi_vis_cmd.txt')
    force_download = 0
    remove_header = 1
    scp = SCP()


def run_unzip(tee_zip_path):
    unzip_cmd = 'unzip {}'.format(tee_zip_path)
    unzip_cmd_list = unzip_cmd.split(' ')
    print('Running {}'.format(unzip_cmd))
    subprocess.check_call(unzip_cmd_list)


def run_scp(params, server_name, log_dir, log_fname, out_dir, is_file, timestamp):
    """

    :param Params.SCP params:
    :param server_name:
    :param dst_rel_path:
    :param is_file:
    :return:
    """

    assert params.pwd and params.user and params.url, "auth data not provided"

    print('\n' + server_name)

    if server_name == 'grs':
        server_home = params.home_dir
    elif server_name == 'x99':
        server_home = linux_path(params.home_dir, "samba_x99")
    elif server_name == 'orca':
        server_home = linux_path(params.home_dir, "samba_orca")
    else:
        raise AssertionError('invalid server name: {}'.format(server_name))

    src_root_path = linux_path(server_home, params.code_path, log_dir)

    zip_fname = "consolidate_log_{}_{}.zip".format(server_name, timestamp)
    zip_path = linux_path(params.home_dir, zip_fname)

    remote_cmd = "cd {} && zip -r {} {}".format(src_root_path, zip_path, log_fname)

    if params.rename_src:
        # log_fname_abs = linux_path(params.home_dir, params.code_path, log_fname)
        # dst_fname_abs = log_fname_abs + '.' + timestamp
        # rename_cmd_abs = 'mv {} {}'.format(log_fname_abs, dst_fname_abs)
        # print('\n' + rename_cmd_abs + '\n')

        log_fname_rel = linux_path(log_dir, log_fname)
        dst_fname_rel = log_fname_rel + '.' + timestamp
        rename_cmd_abs = 'mv {} {}'.format(log_fname_rel, dst_fname_rel)
        print('\n' + rename_cmd_abs + '\n')

        # dst_log_fname = log_fname + '.' + timestamp
        # rename_cmd = 'mv {} {}'.format(log_fname, dst_log_fname)
        # print('\n' + rename_cmd + '\n')

        # print('renaming source to {}'.format(dst_zip_path))
        # remote_cmd = '{} && {}'.format(remote_cmd, rename_cmd)

    if params.show_output == 2:
        print('running {}'.format(remote_cmd))

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(params.url, username=params.user, password=params.pwd)

    stdin, stdout, stderr = client.exec_command(remote_cmd)
    stdout = list(stdout)
    stderr = list(stderr)

    if stderr:
        # for line in stderr:
        #     print(line.strip('\n'))
        raise AssertionError('remote command did not work' + '\n' + '\n'.join(stderr))

    if params.show_output:
        for line in stdout:
            print(line.strip('\n'))

    client.close()

    scp_cmd = "pscp -pw {} -r -P 22 {}@{}:{} ./".format(params.pwd, params.user, params.url, zip_path)
    if params.show_output == 2:
        print('Running {}'.format(scp_cmd))

    scp_cmd_list = scp_cmd.split(' ')
    subprocess.check_call(scp_cmd_list)

    with ZipFile(zip_fname, 'r') as zipObj:
        zipObj.extractall()

    if params.remove_zip:
        subprocess.check_call(['rm', zip_fname])

    out_path = linux_path(out_dir, add_suffix(log_fname, '{}_{}'.format(server_name, timestamp)))

    # if not is_file:
    #     os.makedirs(out_path, exist_ok=True)
    # else:
    #     os.makedirs(os.path.dirname(out_path), exist_ok=True)

    shutil.move(log_fname, out_path)
    return out_path


def main():
    params = Params()

    paramparse.process(params, allow_unknown=1)

    params.scp.read_auth()

    if params.working_dir:
        os.chdir(params.working_dir)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")

    log_data_all = []
    log_data_dict = {}
    out_dir = linux_path(params.log_dir, 'consolidate_log')

    os.makedirs(out_dir, exist_ok=True)

    for server_id, server_name in enumerate(params.servers):
        log_path = run_scp(params.scp, server_name, params.log_dir, params.log_fname, out_dir, is_file=1,
                           timestamp=timestamp)

        log_data = open(log_path, 'r').readlines()

        if server_id > 0 and params.remove_header:
            log_data = log_data[1:]

        log_data_all += log_data

        log_data_dict[server_name] = log_data

    out_fname = add_suffix(params.log_fname, '{}'.format(timestamp))
    out_path = linux_path(out_dir, out_fname)

    print('writing consolidated log to {}'.format(out_path))
    os.makedirs(params.log_dir, exist_ok=True)
    with open(out_path, 'w') as fid:
        fid.write(''.join(log_data_all))


if __name__ == '__main__':
    main()
