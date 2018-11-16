import os
import numpy as np

if __name__ == '__main__':

    # reinit_40_5
    # reinit_0.90_5
    # reinit_0.35_5
    # reinit_0.60_5


    in_dir = './C++/MTF/log/success_rates_txt/success_rates'
    out_dir = './C++/MTF/log/success_rates_txt/bin'
    # in_dir = './C++/MTF/log/success_rates'
    # out_dir = './C++/MTF/log/success_rates'

    # list_fname = None
    list_fname = 'list.txt'

    if not os.path.exists(out_dir):
        print 'Output directory: {:s} does not exist. Creating it...'.format(out_dir)
        os.makedirs(out_dir)

    if list_fname is None:
        file_list = [file for file in os.listdir(in_dir) if file.endswith(".txt")]
    else:
        file_list = open('{:s}/{:s}'.format(in_dir, list_fname), 'r').readlines()
    n_files = len(file_list)
    file_id = 1
    for file in file_list:
        file = file.rstrip()
        in_file_path = '{:s}/{:s}'.format(in_dir, file)
        print 'Processing file {:d} of {:d}: {:s}'.format(file_id, n_files, file)
        try:
            file_data = np.loadtxt(in_file_path, dtype=np.float64)
        except ValueError:
            print 'This file does not contain correctly formatted data'
            continue
        in_file_name = os.path.splitext(file)[0]
        out_file_name = '{:s}.bin'.format(in_file_name)
        out_file_path = '{:s}/{:s}'.format(out_dir, out_file_name)
        out_file = open(out_file_path, 'wb')
        np.array(file_data.shape, dtype=np.uint32).tofile(out_file)
        file_data.tofile(out_file)
        out_file.close()
        file_id += 1