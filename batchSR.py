import os
import zipfile
import sys
import subprocess
from Misc import getParamDict
from Misc import getSyntheticSeqName

if __name__ == '__main__':

    use_arch = 1
    arch_root_dir = './C++/MTF/log/archives'
    arch_name = 'resh_rklcv_10_20_40r_10ki25ic_ssd_50r_30i_4u_8_subseq10_mcd_tulp'
    in_arch_path = 'tracking_data'
    gt_root_dir = '../Datasets'
    tracking_root_dir = './C++/MTF/log/archives'
    # tracking_root_dir = './C++/MTF/log/tracking_data'
    out_dir = './C++/MTF/log/success_rates'
    # list_fname = 'list.txt'
    list_fname = None
    list_in_arch = 0
    # list_fname = '{:s}/{:s}.txt'.format(arch_root_dir, arch_name)
    actor_ids = [0, 1, 2, 3]
    # actor_ids = [15]

    # use_reinit_gt = 1
    # opt_gt_ssms = None
    use_reinit_gt = 0
    opt_gt_ssms = ['0']

    enable_subseq = 1
    reinit_on_failure = 1

    n_runs = 1
    n_subseq = 10
    err_type = 0
    reinit_frame_skip = 5
    jaccard_err_thresh = 0.90
    mcd_err_thresh = 20.0

    err_min = 0
    err_res = 100
    write_err = 0
    overriding_seq_id = -1

    reinit_at_each_frame = 0
    reset_at_each_frame = 0
    reset_to_init = 1

    # settings for synthetic sequences
    syn_ssm = 'c8'
    syn_ssm_sigma_ids = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    # syn_ssm_sigma_ids = [94, 95, 96, 97, 98, 99, 100, 101, 102, 103]
    syn_ilm = '0'
    syn_am_sigma_ids = [9]
    syn_add_noise = 0
    syn_noise_mean = 0
    syn_noise_sigma = 10
    syn_frame_id = 0
    syn_err_thresh = 5.0

    write_to_bin = 1

    arg_id = 1
    if len(sys.argv) > arg_id:
        arch_name = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        reinit_on_failure = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        reinit_frame_skip = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        reinit_err_thresh = float(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        reinit_at_each_frame = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        arch_root_dir = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        gt_root_dir = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        tracking_root_dir = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        out_dir = sys.argv[arg_id]
        arg_id += 1

    params_dict = getParamDict()
    actors = params_dict['actors']
    sequences = params_dict['sequences']

    if actors[actor_ids[0]] == 'Synthetic' and len(actor_ids) == 1:
        enable_subseq = 0
        mcd_err_thresh = syn_err_thresh

    if err_type == 2:
        # Jaccard error
        reinit_err_thresh = jaccard_err_thresh
        err_max = jaccard_err_thresh
        overflow_err = 1e3
    else:
        # MCD/CL error
        reinit_err_thresh = mcd_err_thresh
        err_max = mcd_err_thresh
        overflow_err = 1e3

    # reinit gt only used with reinit tests
    # use_reinit_gt = use_reinit_gt or reinit_from_gt

    if reinit_at_each_frame or reset_at_each_frame:
        reinit_on_failure = 0
    # sub sequence tests only run without reinitialization
    enable_subseq = enable_subseq and not reinit_on_failure and not reinit_at_each_frame and not reset_at_each_frame

    if use_arch:
        arch_path = '{:s}/{:s}.zip'.format(arch_root_dir, arch_name)
        print 'Reading tracking data from zip archive: {:s}'.format(arch_path)
        arch_fid = zipfile.ZipFile(arch_path, 'r')
    else:
        data_path = '{:s}/{:s}'.format(arch_root_dir, arch_name)
        tracking_root_dir = '{:s}/{:s}'.format(data_path, in_arch_path)

    # if not os.path.isfile(list_fname):
    # print 'List file for the batch job does  not exist:\n {:s}'.format(list_fname)

    file_list = None
    if list_fname is not None:
        if list_in_arch:
            if use_arch:
                file_list = arch_fid.open(list_fname, 'r').readlines()
            else:
                file_list = open('{:s}/{:s}'.format(data_path, list_fname), 'r').readlines()
        else:
            file_list = open('{:s}/{:s}'.format(arch_root_dir, list_fname), 'r').readlines()

            # exclude files corresponding to sub sequence runs if any
        file_list = [file_name for file_name in file_list
                     if '_init_' not in file_name and '.txt' in file_name]
        # file_list = [file_name for file_name in file_list if '.txt' in file_name]
        n_files = len(file_list)
        if opt_gt_ssms is not None and len(opt_gt_ssms) > 1 and len(opt_gt_ssms) != n_files:
            raise SyntaxError('Incorrect number of optimal GT specifiers given: {:d}'.format(len(opt_gt_ssms)))
        print 'Generating success rates for following {:d} files for all actors: \n'.format(n_files), file_list

    for actor_id in actor_ids:
        actor = actors[actor_id]
        if actor == 'Synthetic':
            reset_at_each_frame = 1
        if list_fname is None or file_list is None:
            if overriding_seq_id >= 0:
                seq_name = sequences[actor][overriding_seq_id]
            else:
                seq_name = sequences[actor][0]
            if actor == 'Synthetic':
                reset_at_each_frame = 1
                seq_name = getSyntheticSeqName(seq_name, syn_ssm, syn_ssm_sigma_ids[0], syn_ilm,
                                               syn_am_sigma_ids[0], syn_frame_id, syn_add_noise,
                                               syn_noise_mean, syn_noise_sigma)
            if reinit_at_each_frame:
                proc_file_path = '{:s}/reinit/{:s}/{:s}'.format(in_arch_path, actor, seq_name)
            elif reset_at_each_frame:
                if reset_to_init:
                    proc_file_path = '{:s}/reset_to_init/{:s}/{:s}'.format(in_arch_path, actor, seq_name)
                else:
                    proc_file_path = '{:s}/reset/{:s}/{:s}'.format(in_arch_path, actor, seq_name)
            elif reinit_on_failure:
                if reinit_err_thresh == int(reinit_err_thresh):
                    proc_file_path = '{:s}/reinit_{:d}_{:d}/{:s}/{:s}'.format(
                        in_arch_path, int(reinit_err_thresh), reinit_frame_skip, actor, seq_name)
                else:
                    proc_file_path = '{:s}/reinit_{:4.2f}_{:d}/{:s}/{:s}'.format(
                        in_arch_path, reinit_err_thresh, reinit_frame_skip, actor, seq_name)
            else:
                proc_file_path = '{:s}/{:s}/{:s}'.format(in_arch_path, actor, seq_name)

            if use_arch:
                path_list = [f for f in arch_fid.namelist() if f.startswith(proc_file_path)]
            else:
                proc_file_path = '{:s}/{:s}'.format(data_path, proc_file_path)
                # print os.listdir(data_path)
                path_list = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_path)] for val in sublist]
                path_list = [f for f in [j.replace('\\', '/') for j in path_list] if f.startswith(proc_file_path)]
                # print '\n'.join(path_list)
                # print proc_file_path
                # exit(0)

            file_list = [os.path.basename(path) for path in path_list]
            file_list = [file_name for file_name in file_list
                         if '_init_' not in file_name and '.txt' in file_name]
            n_files = len(file_list)
            if opt_gt_ssms is not None and len(opt_gt_ssms) > 1 and len(opt_gt_ssms) != n_files:
                raise SyntaxError('Incorrect number of optimal GT specifiers given: {:d}'.format(len(opt_gt_ssms)))
            print 'Generating success rates for following {:d} files for actor {:d}: \n'.format(n_files,
                                                                                                actor_id), file_list
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
            if opt_gt_ssms is None:
                opt_gt_ssm = mtf_ssm
                if opt_gt_ssm == '8' or opt_gt_ssm == 'c8' or opt_gt_ssm == 'l8' or opt_gt_ssm == 'sl3':
                    opt_gt_ssm = '0'
            elif len(opt_gt_ssms) > 1:
                opt_gt_ssm = opt_gt_ssms[line_id]
            else:
                opt_gt_ssm = opt_gt_ssms[0]

            arguments = '{:d} {:s} {:s} {:s} {:s}'.format(
                actor_id, mtf_sm, mtf_am, mtf_ssm, iiw)
            arguments = '{:s} {:d} {:s} {:s} {:s} {:s} {:s} {:s}'.format(
                arguments, use_arch, arch_name, in_arch_path, arch_root_dir, gt_root_dir,
                tracking_root_dir, out_dir)
            arguments = '{:s} {:s} {:d} {:d} {:d} {:f} {:d} {:d} {:d} {:d} {:d} {:d}'.format(
                arguments, opt_gt_ssm, use_reinit_gt, reinit_on_failure, reinit_frame_skip,
                reinit_err_thresh, reinit_at_each_frame, reset_at_each_frame, reset_to_init,
                enable_subseq, n_subseq, overriding_seq_id)
            arguments = '{:s} {:f} {:f} {:d} {:d} {:d} {:f} {:d}'.format(
                arguments, err_min, err_max, err_res, err_type, write_err,
                overflow_err, write_to_bin)
            if n_runs > 1:
                arguments = '{:s} {:d}'.format(arguments, n_runs)
                full_command = 'python successGeneralFastMR.py {:s}'.format(arguments)
            else:
                full_command = 'python successGeneralFast.py {:s}'.format(arguments)

            # full_command = \
            # 'python successGeneralFast.py {:d} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:d} {:d} {:d} {:f} {:d} {:d} {:f} {:f} {:d} {:d} {:d} {:f}'.format(
            # actor_id, mtf_sm, mtf_am, mtf_ssm, iiw, arch_name, in_arch_path, arch_root_dir, gt_root_dir,
            # tracking_root_dir, out_dir, opt_gt_ssm, use_reinit_gt, reinit_from_gt, reinit_frame_skip,
            # reinit_err_thresh, enable_subseq, n_subseq, err_min, err_max, err_res, err_type, write_err,
            # overflow_err)

            if actor == 'Synthetic':
                for syn_ssm_sigma_id in syn_ssm_sigma_ids:
                    for syn_am_sigma_id in syn_am_sigma_ids:
                        syn_arguments = '{:s} {:d} {:s} {:d} {:d} {:d} {:d} {:d}'.format(
                            syn_ssm, syn_ssm_sigma_id, syn_ilm, syn_am_sigma_id, syn_add_noise, syn_noise_mean,
                            syn_noise_sigma, syn_frame_id)
                        syn_full_command = '{:s} {:s}'.format(full_command, syn_arguments)
                        print 'running: {:s}'.format(syn_full_command)
                        subprocess.check_call(syn_full_command, shell=True)
            else:
                print 'running: {:s}'.format(full_command)
                subprocess.check_call(full_command, shell=True)
            # status = os.system(full_command)
            # if not status:
            # s = raw_input('Last command not completed successfully. Continue ?\n')
            # if s == 'n' or s == 'N':
            # sys.exit()
            line_id += 1