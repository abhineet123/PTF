import subprocess
import re


def get_interfaces():
    output = subprocess.check_output("ipconfig /all")



    lines = output.splitlines()
    lines = filter(lambda x: x, lines)

    # print('output: ', output)
    # print('lines: ', lines)

    ip_address = ''
    # mac_address = ''
    name = ''

    for line in lines:
        # -------------
        # Interface Name

        is_interface_name = re.match(r'^[a-zA-Z0-9].*:$', line)
        # is_interface_name = 1
        if is_interface_name:

            # Check if there's previews values, if so - yield them
            if name and ip_address:
                yield {
                    "ip_address": ip_address,
                    # "mac_address": mac_address,
                    "name": name,
                }

            ip_address = ''
            # mac_address = ''
            name = line.rstrip(':')

            # print('line: ', line)
            # print('name: ', name)

        line = line.strip().lower()

        if ':' not in line:
            continue

        value = line.split(':')[-1]
        value = value.strip()

        # -------------
        # IP Address

        is_ip_address = not ip_address and re.match(r'ipv4 address|autoconfiguration ipv4 address|ip address', line)

        if is_ip_address:
            ip_address = value
            ip_address = ip_address.replace('(preferred)', '')
            ip_address = ip_address.strip()

            # print('line: ', line)
            # print('ip_address: ', ip_address)

        # -------------
        # MAC Address

        # is_mac_address = not ip_address and re.match(r'physical address', line)
        #
        # if is_mac_address:
        #     mac_address = value
        #     mac_address = mac_address.replace('-', ':')
        #     mac_address = mac_address.strip()

    if name and ip_address:
        yield {
            "ip_address": ip_address,
            # "mac_address": mac_address,
            "name": name,
        }


if __name__ == '__main__':
    for interface in get_interfaces():
        print interface