import os
import zipfile
import sys
import subprocess

if __name__ == '__main__':

    use_arch = 1
    arch_root_dir = './C++/MTF/log/archives/CRV'
    arch_name = 'ssd_like__nn5k10k'
    in_arch_path = 'tracking_data'
    gt_root_dir = '../Datasets'
    tracking_root_dir = './C++/MTF/log/tracking_data'
    out_dir = './C++/MTF/log/recovery_rates'

    actor_ids = [0, 1]
    opt_gt_ssms = ['0']
    reinit_from_gt = 0
    reinit_frame_skip = 5
    reinit_err_thresh = 20
    list_fname = 'list.txt'
    # list_fname = '{:s}/{:s}.txt'.format(arch_root_dir, arch_name)

    arch_path = '{:s}/{:s}.zip'.format(arch_root_dir, arch_name)
    print 'Reading tracking data from zip archive: {:s}'.format(arch_path)
    arch_fid = zipfile.ZipFile(arch_path, 'r')

    # if not os.path.isfile(list_fname):
    # print 'List file for the batch job does  not exist:\n {:s}'.format(list_fname)

    file_list = arch_fid.open(list_fname, 'r').readlines()
    n_files = len(file_list)
    if len(opt_gt_ssms) > 1 and len(opt_gt_ssms) != n_files:
        raise SyntaxError('Incorrect number of optimal GT specifiers given: {:d}'.format(len(opt_gt_ssms)))
    print 'Generating success rates for following {:d} files: \n'.format(n_files), file_list
    line_id = 0
    for line in file_list:
        line = line.rstrip()
        # line = line.rstrip('.txt')
        line = os.path.splitext(line)[0]
        print 'processing line: ', line
        words = line.split('_')
        mtf_sm = words[0]
        mtf_am = words[1]
        mtf_ssm = words[2]
        iiw = words[3]
        if (len(opt_gt_ssms) > 1):
            opt_gt_ssm = opt_gt_ssms[line_id]
        else:
            opt_gt_ssm = opt_gt_ssms[0]
        for actor_id in actor_ids:
            full_command =\
                'python recoveryGeneralFast.py {:d} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:d} {:d} {:f}'.format(
                actor_id, mtf_sm, mtf_am, mtf_ssm, iiw, arch_name, in_arch_path, arch_root_dir, gt_root_dir, tracking_root_dir,
                out_dir, opt_gt_ssm, reinit_from_gt, reinit_frame_skip, reinit_err_thresh)
            print 'running: {:s}'.format(full_command)
            subprocess.check_call(full_command, shell=True)
            # status = os.system(full_command)
            # if not status:
            # s = raw_input('Last command not completed successfully. Continue ?\n')
            # if s == 'n' or s == 'N':
            # sys.exit()
        line_id += 1


