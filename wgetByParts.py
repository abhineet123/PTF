import paramparse
import math
import os


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
    if not params.size:
        raise IOError('Size must be provided')

    if not params.n_parts and not params.part_size:
        raise IOError('Either n_parts or part_size must be provided')

    if params.n_parts:
        part_size = float(params.size) / float(params.n_parts)
        n_parts = params.n_parts
    elif params.part_size:
        part_size = params.part_size
        n_parts = math.ceil(params.size / part_size)

    print('Downloading file of size {} GB from {} in {} parts of size {} each'.format(
        params.size, params.url, n_parts, part_size
    ))

    start_range = 0
    for i in range(params.n_parts):
        end_range = start_range + part_size
        start_range_b = int(start_range*1e9)
        end_range_b = int(end_range*1e9)

        print('Downloading part {} with range {} - {} GB'.format(i+1, start_range, end_range))

        curl_cmd = 'curl --range {}-{} -o {}.part{} {}'.format(
            start_range_b, end_range_b, params.out_name, i+1, params.url)

        print('Running command: {}'.format(curl_cmd))
        os.system(curl_cmd)

