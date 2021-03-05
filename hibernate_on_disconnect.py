import subprocess
import time
import sys
import os
import re
from datetime import datetime

from Misc import processArguments


def is_connected(interface_name):
    output = subprocess.check_output("ipconfig /all")

    lines = output.splitlines()
    lines = filter(lambda x: x, lines)

    lines = map(str, lines)

    # print('interface_name: {}'.format(interface_name))

    # print('output: {}'.format(output))

    name = None

    for line in lines:

        line = str(line.strip().lower())

        # print('line: {}'.format(line))

        is_interface_name = re.match(r'^[a-zA-Z0-9].*:$', line)
        # is_interface_name = line.startswith('ethernet adapter')

        # print('is_interface_name: {}'.format(is_interface_name))

        if is_interface_name:
            # name = line.replace('ethernet adapter', '').rstrip(':').strip()
            name = line.rstrip(':').strip()

            # print('\n\nname: {}'.format(name))

            continue

        if ':' not in line:
            continue

        value = line.split(':')[-1]
        value = value.strip()

        is_media_state = line.startswith('media state')

        # print('is_media_state: {}'.format(is_media_state))

        if is_media_state:
            media_state = value

            # print('media_state: {}'.format(media_state))
            is_disconnected = media_state == 'media disconnected'
            is_target_interface = name == interface_name

            # print('is_disconnected: {}'.format(is_disconnected))
            # print('is_target_interface: {}'.format(is_target_interface))
            # print('name: {}'.format(name))
            # print('interface_name: {}'.format(interface_name))

            if is_disconnected and is_target_interface:
                return False

    if name is None:
        raise IOError('interface_name: {} not found'.format(interface_name))

    return True


def main():
    params = {
        'interface_name': 'ethernet adapter ethernet',
        'utorrent_mode': 1,
        'sleep_time': 10,
        'wait_time': 10800,
        'post_wait_time': 10,
        'check_vpn_gap': 30,
        'max_vpn_wait_time': 600,
        'max_disconnects': 5,
        'vpn_path': 'C:/Users/Tommy/Desktop/purevpn.lnk',
        'tor_path': 'C:/Users/Tommy/Desktop/uTorrent.lnk',
        'settings_path': 'C:/Users/Tommy/AppData/Roaming/uTorrent/settings.dat',
        'vpn_proc': 'PureVPN.exe',
        'tor_proc': 'uTorrent.exe',
        'syatem_name': 'GT1K',
        'email_auth': [],
    }

    processArguments(sys.argv[1:], params)

    interface_name = params['interface_name']
    sleep_time = params['sleep_time']
    max_disconnects = params['max_disconnects']

    # hibernate_now = 0

    interface_name = interface_name.lower()

    n_disconnects = 0

    while True:
        if not is_connected(interface_name):
            time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
            n_disconnects += 1
            print("{} :: disconnect {}".format(time_stamp, n_disconnects))
        else:
            n_disconnects = 0
            
        if n_disconnects > max_disconnects:
            # hibernate_now = 1
            time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
            print("{} :: hibernating...".format(time_stamp))
            os.system("shutdown /h")
        try:
            time.sleep(sleep_time)
        except KeyboardInterrupt:
            exit()

    # if hibernate_now:



if __name__ == '__main__':
    main()
