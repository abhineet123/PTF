import zipfile
import numpy as np

if __name__ == '__main__':
    arch_root_dir = './C++/MTF/log/archives'
    out_dir = './C++/MTF/log/error_data'
    arch_list_fname = 'arch_list.txt'
    arch_names = open('{:s}/{:s}'.format(arch_root_dir, arch_list_fname), 'r').readlines()
    total_n_error = 0
    print 'reading following {:d} archives: '.format(len(arch_names)), arch_names
    for arch_name in arch_names:
        arch_path = '{:s}/{:s}'.format(arch_root_dir, arch_name).strip()
        print 'Reading error data from zip archive: {:s}'.format(arch_path)
        arch_fid = zipfile.ZipFile(arch_path, 'r')
        path_list = [f for f in arch_fid.namelist() if '.err' in f]
        n_error = 0
        collected_error_data = []
        for path in path_list:
            error_data_lines = arch_fid.open(path, 'r').readlines()
            # remove header
            del (error_data_lines[0])
            n_error += len(error_data_lines)
            curr_err_data = []
            for line in error_data_lines:
                line_data = line.split()
                curr_err_data.append(
                    [float(line_data[1]),
                     float(line_data[2]),
                     float(line_data[3])]
                )
            collected_error_data.extend(curr_err_data)
        total_n_error += n_error
        print 'read {:d} errors'.format(n_error)
        if n_error > 0:
            out_path = '{:s}/{:s}'.format(out_dir, arch_name.replace('.zip', '.err').strip())
            print 'Saving error data to {:s}'.format(out_path)
            np.savetxt(out_path, np.array(collected_error_data), fmt='%15.9f', delimiter='\t')
    print 'read a total of {:d} errors'.format(total_n_error)






