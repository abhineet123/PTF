import os
import shutil

if __name__ == '__main__':

    in_dir = './C++/MTF/log/success_rates/TMT'
    out_dir = './C++/MTF/log/success_rates/temp'
    file_ext = 'bin'

    if not os.path.exists(out_dir):
        print 'Output directory: {:s} does not exist. Creating it...'.format(out_dir)
        os.makedirs(out_dir)

    file_list = [file for file in os.listdir(in_dir) if file.endswith('.{:s}'.format(file_ext))]

    files_found = []

    for file in file_list:
        file_ss = file.replace('.{:s}'.format(file_ext), '_subseq_10.{:s}'.format(file_ext))


        if file_ss in file_list:
            print 'file: {:s} file_ss: {:s}'.format(file, file_ss)
            files_found.append(file)
            shutil.move(os.path.join(in_dir, file), os.path.join(out_dir, file))


    print 'Total files found: {:d}'.format(len(files_found))
    print 'Total files searched: {:d}'.format(len(file_list))





