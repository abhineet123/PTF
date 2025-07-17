import os
import time

import paramiko
import paramparse

from win10toast import ToastNotifier
import winsound

from datetime import datetime

from Misc import linux_path


class Params(paramparse.CFG):
    duration = 1000
    n_times = 4
    freq = 440  # Hz
    info_file = ''
    remote = ''
    proxy = ''
    sleep_time = 5


def read_remote_info(info_file):
    info_path = linux_path(os.path.expanduser('~'), info_file)

    info_data = open(info_path, 'r').readlines()
    info_data = [k.strip() for k in info_data]

    dst_info = {}
    for datum in info_data:
        info = datum.split(' ')
        name0, name1, dst = info[:3]
        port = 22
        ecr = key = None
        if len(info) > 3:
            port = int(info[3])
        if len(info) > 4:
            ecr = info[4]
        if len(info) > 5:
            key = info[5]
        dst_info[name0] = [name0, name1, dst, ecr, key, port]
    return dst_info


def connect_to_remote(info_file, remote, proxy):
    remote_info = read_remote_info(info_file)
    name0, name1, dst, ecr, key, port = remote_info[remote]

    username, server = dst.split('@')
    print(f'connecting to remote {remote}')

    ssh = paramiko.SSHClient()

    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(server, username=username, port=port, timeout=5)
    except paramiko.ssh_exception.NoValidConnectionsError:
        print(f'unable to connect to {server}')
        return None
    except BaseException:
        print(f'unable to connect to {server}')
        return None


    ssh_main = None

    if proxy:
        print(f'connecting to proxy remote {proxy}')

        _, _, proxy_dst, _, _, proxy_port = remote_info[proxy]
        proxy_username, proxy_server = proxy_dst.split('@')

        vmtransport = ssh.get_transport()
        dest_addr = (proxy_server, proxy_port)  # edited#
        local_addr = (server, port)  # edited#
        try:
            vmchannel = vmtransport.open_channel(
                "direct-tcpip", dest_addr, local_addr)
        except paramiko.ssh_exception.ChannelException as e:
            print(f'\n\nfailed to open_channel:\n{e}\n')
            return None

        proxy_ssh = paramiko.SSHClient()
        proxy_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            proxy_ssh.connect(proxy_server, username=proxy_username, sock=vmchannel)
        except paramiko.ssh_exception.NoValidConnectionsError:
            print(f'unable to connect to {proxy_server}')
            return None
        except BaseException:
            print(f'unable to connect to {proxy_server}')
            return None

        ssh_main = ssh
        ssh = proxy_ssh
    return ssh, ssh_main


def main(params: Params, toast: ToastNotifier):
    ret = connect_to_remote(params.info_file, params.remote, params.proxy)
    # status_file = 'server_connect_success.txt'
    if ret is not None:
        print('success')
        # if os.path.exists(status_file):
        #     time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        #     open(status_file, 'w').write(time_stamp)
    else:
        toast.show_toast(
            "Server connection failed",
            "Server connection failed",
            duration=20,
            # icon_path="icon.ico",
            threaded=True,
        )
        for _ in range(params.n_times):
            winsound.Beep(params.freq, params.duration)
            # os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        # print('failure')
        # if os.path.exists(status_file):
        #     os.remove(status_file)


if __name__ == '__main__':
    params: Params = paramparse.process(Params)
    toast = ToastNotifier()

    while True:
        try:
            main(params, toast)
            time.sleep(params.sleep_time)
        except KeyboardInterrupt:
            break
