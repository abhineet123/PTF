import subprocess
import time, os, sys

from Misc import processArguments


def is_connected(interface_name='ethernet'):
    output = subprocess.check_output("ipconfig /all")

    lines = output.splitlines()
    lines = filter(lambda x: x, lines)

    lines = map(str, lines)

    print('interface_name: {}'.format(interface_name))

    print('output: {}'.format(output))

    name = None

    for line in lines:

        line = str(line.strip().lower())

        print('line: {}'.format(line))

        is_interface_name = line.startswith('ethernet adapter')

        print('is_interface_name: {}'.format(is_interface_name))

        if is_interface_name:
            name = line.replace('ethernet adapter', '').rstrip(':').strip()

            print('\n\nname: {}'.format(name))

            continue

        if ':' not in line:
            continue

        value = line.split(':')[-1]
        value = value.strip()

        is_media_state = line.startswith('media state')

        print('is_media_state: {}'.format(is_media_state))

        if is_media_state:
            media_state = value

            print('media_state: {}'.format(media_state))
            is_disconnected = media_state == 'media disconnected'
            is_target_interface = name == interface_name

            print('is_disconnected: {}'.format(is_disconnected))
            print('is_target_interface: {}'.format(is_target_interface))
            print('name: {}'.format(name))
            print('interface_name: {}'.format(interface_name))

            if is_disconnected and is_target_interface:
                return False

    if name is None:
        raise IOError('interface_name: {} not found'.format(interface_name))

    return True

if __name__ == '__main__':

    params = {
        'interface_name': 'ethernet',
        'utorrent_mode': 1,
        'restart_time': 86400,
        'wait_time': 10800,
        'post_wait_time': 10,
        'check_vpn_gap': 30,
        'max_vpn_wait_time': 600,
        'proc_kill_type': 0,
        'vpn_path': 'C:/Users/Tommy/Desktop/purevpn.lnk',
        'tor_path': 'C:/Users/Tommy/Desktop/uTorrent.lnk',
        'settings_path': 'C:/Users/Tommy/AppData/Roaming/uTorrent/settings.dat',
        'vpn_proc': 'PureVPN.exe',
        'tor_proc': 'uTorrent.exe',
        'syatem_name': 'GT1K',
        'email_auth': [],
    }

    # paramparse.process_dict(params)

    processArguments(sys.argv[1:], params)

    utorrent_mode = params['utorrent_mode']
    interface_name = params['interface_name']
    restart_time = params['restart_time']
    wait_time = params['wait_time']
    post_wait_time = params['post_wait_time']
    check_vpn_gap = params['check_vpn_gap']
    vpn_path = params['vpn_path']
    tor_path = params['tor_path']
    settings_path = params['settings_path']
    vpn_proc = params['vpn_proc']
    tor_proc = params['tor_proc']
    email_auth = params['email_auth']
    max_vpn_wait_time = params['max_vpn_wait_time']
    proc_kill_type = params['proc_kill_type']

    global_start_t = time.time()

    hibernate_now = 0

    interface_name = interface_name.lower()

    while True:
        if not is_connected(interface_name):
            hibernate_now = 1
            break
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            exit()

    if hibernate_now:
        print("hibernating...")
        # os.system("shutdown /h")
