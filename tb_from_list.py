import os

import paramparse


class Params:
    def __init__(self):
        self.cfg = ()
        self.end_id = -1
        self.images = 100
        self.list_fname = 'tb.cfg'
        self.python_exe = 'python36'
        self.start_id = 0
        self.tb_path = '/usr/local/lib/python3.6/dist-packages/tensorboard/main.py'


def main():
    params = Params()
    paramparse.process(params)

    list_fname = params.list_fname

    while True:
        if os.path.isfile(list_fname):
            print('reading tb log dir list from: {}'.format(list_fname))
            data = open(list_fname, 'r').readlines()

            data_list = [k.strip().split('\t') for k in data]

            if params.start_id > 0:
                data_list = data_list[params.start_id:]

            data_list_str = ['{}:{}'.format(k[0], k[1]) for k in data_list if os.path.isdir(k[1])]
            log_dirs = ','.join(data_list_str)
            log_dirs_arg = '--logdir_spec'

        else:
            log_dirs = list_fname

            log_dirs_arg = '--logdir'

        tb_cmd = "{} {} {}={} --bind_all --samples_per_plugin images={}".format(
            params.python_exe, params.tb_path, log_dirs_arg, log_dirs, params.images,
        )

        print('running: {}'.format(tb_cmd))

        os.system(tb_cmd)

        list_fname = input('\nEnter log folder / list file name\n')

        if not list_fname.strip():
            list_fname = params.list_fname


if __name__ == '__main__':
    main()
