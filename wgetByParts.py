import paramparse
import math
import os
import subprocess


class WgetByPartsParams:
    """
    :param str url:
    :param float size:
    :param int n_parts:
    :param float part_size:
    """

    def __init__(self):
        self.url = ''
        self.out_name = 'wgetp_file'
        self.size = 0.0
        self.n_parts = 0
        self.part_size = 0.0


if __name__ == '__main__':
    params = WgetByPartsParams()
    paramparse.process(params)

    if not params.url:
        raise IOError('URL must be provided')

    if not params.n_parts and not params.part_size:
        raise IOError('Either n_parts or part_size must be provided')

    if not params.size:
        print('Attempting to get size using curl')
        curl_cmd = "curl -sI {}  | grep -i Content-Length | awk '{print $2}'"
        size_output = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE).communicate()[0]
        try:
            size = int(size_output)
        except BaseException as e:
            raise IOError('Failed to get size : {} :: {}'.format(e, size_output))
        size = float(size) / 1e9
    else:
        size = params.size

    if params.n_parts:
        part_size = size / params.n_parts
        n_parts = params.n_parts
    elif params.part_size:
        part_size = params.part_size
        n_parts = math.ceil(size / part_size)

    print('Downloading file of size {} GB from {} in {} parts of size {} each'.format(
        size, params.url, n_parts, part_size
    ))

    start_range = 0
    for i in range(params.n_parts):
        end_range = start_range + part_size

        if i == params.n_parts - 1:
            end_range_str = ''
        else:
            end_range_str = '{}'.format(int(end_range * 1e9))

        start_range_str = '{}'.format(int(start_range * 1e9))
        print('Downloading part {} with range {} - {} GB'.format(i + 1, start_range, end_range))

        curl_cmd = 'curl --range {}-{} -o {}.part{} {}'.format(
            start_range_str, end_range_str, params.out_name, i + 1, params.url)

        print('Running command: {}'.format(curl_cmd))
        # os.system(curl_cmd)

        start_range = end_range + 1