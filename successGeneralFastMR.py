from Misc import getParamDict
from Misc import readDistGridParams
from Misc import getTrackingErrors
from Misc import getSyntheticSeqName
from Misc import getSyntheticSeqSuffix

import itertools as it

import numpy as np
import sys
import os
import zipfile

if __name__ == '__main__':
    use_arch = 1
    arch_root_dir = './C++/MTF/log/archives'
    arch_name = 'ncc__esm__3_4_6__lintrack'
    in_arch_path = 'tracking_data'
    tracking_root_dir = './C++/MTF/log/tracking_data'
    gt_root_dir = '../Datasets'
    out_dir = './C++/MTF/log/success_rates'
    err_out_dir = './C++/MTF/log/tracking_errors'
    # set this to '0' to use the original ground truth
    mtf_sm = 'ialkC1'
    mtf_am = 'rscv50r30i4u'
    mtf_ssm = '8'
    iiw = 1
    opt_gt_ssm = '0'
    use_reinit_gt = 1
    write_to_bin = 0
    n_runs = 1

    reinit_on_failure = 0
    reinit_frame_skip = 5
    reinit_err_thresh = 20
    reinit_at_each_frame = 0
    # mtf_am = 'miIN50r10i8b'
    # mtf_am = 'miIN50r10i1i5000s'
    enable_subseq = 0
    n_subseq = 10

    err_min = 0
    err_max = 20.0
    err_res = 100
    err_type = 0

    overflow_err = 1e3
    write_err = 1

    show_jaccard_img = 0

    # settings for synthetic sequences
    syn_ssm = 'c8'
    syn_ssm_sigma_id = 19
    syn_ilm = '0'
    syn_am_sigma_id = 0
    syn_add_noise = 0
    syn_noise_mean = 0
    syn_noise_sigma = 10
    syn_frame_id = 0

    params_dict = getParamDict()
    param_ids = readDistGridParams()

    actors = params_dict['actors']
    sequences = params_dict['sequences']
    mtf_sms = params_dict['mtf_sms']
    mtf_ams = params_dict['mtf_ams']
    mtf_ssms = params_dict['mtf_ssms']

    actor_id = param_ids['actor_id']
    mtf_sm_id = param_ids['mtf_sm_id']
    mtf_am_id = param_ids['mtf_am_id']
    mtf_ssm_id = param_ids['mtf_ssm_id']

    overriding_seq_id = -1
    # iiw = param_ids['init_identity_warp']

    arg_id = 1
    if len(sys.argv) > arg_id:
        actor_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        mtf_sm = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        mtf_am = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        mtf_ssm = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        iiw = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        use_arch = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        arch_name = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        in_arch_path = sys.argv[arg_id]
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
    if len(sys.argv) > arg_id:
        opt_gt_ssm = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        use_reinit_gt = int(sys.argv[arg_id])
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
        reinit_at_each_frame = float(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        enable_subseq = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        n_subseq = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        overriding_seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        err_min = float(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        err_max = float(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        err_res = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        err_type = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        write_err = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        overflow_err = float(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        write_to_bin = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        n_runs = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_ssm = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_ssm_sigma_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_ilm = sys.argv[arg_id]
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_am_sigma_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_add_noise = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_noise_mean = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_noise_sigma = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        syn_frame_id = int(sys.argv[arg_id])
        arg_id += 1

    if n_runs <= 1:
        raise StandardError('successGeneralFastMR :: no. of runs must be greater than 1')

    use_opt_gt = 0

    if opt_gt_ssm == '0':
        print 'Using standard ground truth'
        use_opt_gt = 0
    else:
        use_opt_gt = 1
        print 'Using optimized ground truth with ssm: ', opt_gt_ssm
        if opt_gt_ssm == '2' or opt_gt_ssm == '2r':
            print 'Using center location error'

    if mtf_sm is None:
        mtf_sm = mtf_sms[mtf_sm_id]
    if mtf_am is None:
        mtf_am = mtf_ams[mtf_am_id]
    if mtf_ssm is None:
        mtf_ssm = mtf_ssms[mtf_ssm_id]
    actor = actors[actor_id]
    sequences = sequences[actor]

    using_pf = False
    if mtf_sm == 'pf':
        using_pf = True

    # err_thresholds = range(1, 20)
    err_thresholds = np.linspace(err_min, err_max, err_res)

    if overriding_seq_id >= 0:
        print 'Generating results for only sequence {:d}: {:s}'.format(
            overriding_seq_id, sequences[overriding_seq_id])
        seq_ids = [overriding_seq_id]
    else:
        seq_ids = range(0, len(sequences))
    # seq_ids = range(2, 3)
    n_err_thr = err_thresholds.size

    n_seq = len(seq_ids)
    # if actor_id == 0:
    # n_seq = 98

    if err_type == 0:
        err_name = 'Mean Corner Distance'
        err_postfix = 'mcd'
    elif err_type == 1:
        err_name = 'Center Location'
        err_postfix = 'cl'
    elif err_type == 2:
        err_name = 'Jaccard'
        err_postfix = 'jaccard'
    else:
        raise StandardError('Invalid error type provided: {:d}'.format(err_type))

    print 'Using {:s} error to evaluate tracking performance'.format(err_name)

    print 'Average trackking errors over {:d} runs'.format(n_runs)

    print 'actor: {:s}'.format(actor)
    print 'mtf_sm: {:s}'.format(mtf_sm)
    print 'mtf_am: {:s}'.format(mtf_am)
    print 'mtf_ssm: {:s}'.format(mtf_ssm)
    print 'iiw: {:d}'.format(iiw)
    print 'use_opt_gt: {:d}'.format(use_opt_gt)
    print 'n_seq: {:d}'.format(n_seq)

    # if os.path.isfile(out_path):
    # s = raw_input('Output file already exists:\n {:s}\n Proceed with overwrite ?\n'.format(out_path))
    # if s == 'n' or s == 'N':
    # sys.exit()

    start_ids = None
    subseq_start_ids = None
    if enable_subseq:
        subseq_file_path = '{:s}/{:s}/subseq_start_ids_{:d}.txt'.format(gt_root_dir, actor, n_subseq)
        print 'Reading sub sequence start frame ids from : {:s}'.format(subseq_file_path)
        subseq_start_ids = np.loadtxt(subseq_file_path, delimiter=',', dtype=np.uint32)
        if subseq_start_ids.shape[0] != len(sequences) or subseq_start_ids.shape[1] != n_subseq:
            raise StandardError('Sub sequence start id file has invalid data size: {:d} x {:d}'.format(
                subseq_start_ids.shape[0], subseq_start_ids.shape[1]))

    arch_fid = None
    if use_arch:
        arch_path = '{:s}/{:s}.zip'.format(arch_root_dir, arch_name)
        print 'Reading tracking data from zip archive: {:s}'.format(arch_path)
        arch_fid = zipfile.ZipFile(arch_path, 'r')
    if use_reinit_gt:
        gt_dir = '{:s}/{:s}/ReinitGT'.format(gt_root_dir, actor)
    else:
        if use_opt_gt:
            print 'Using {:s} SSM GT'.format(opt_gt_ssm)
            gt_dir = '{:s}/{:s}/OptGT'.format(gt_root_dir, actor)
        else:
            gt_dir = '{:s}/{:s}'.format(gt_root_dir, actor)
    print 'Reading GT from folder : {:s}'.format(gt_dir)

    if reinit_at_each_frame:
        in_arch_path = '{:s}/reinit'.format(in_arch_path)
        tracking_root_dir = '{:s}/reinit'.format(tracking_root_dir)
        out_dir = '{:s}/reinit'.format(out_dir)
    elif reinit_on_failure:
        if reinit_err_thresh == int(reinit_err_thresh):
            in_arch_path = '{:s}/reinit_{:d}_{:d}'.format(in_arch_path, int(reinit_err_thresh), reinit_frame_skip)
            tracking_root_dir = '{:s}/reinit_{:d}_{:d}'.format(tracking_root_dir, int(reinit_err_thresh),
                                                               reinit_frame_skip)
            out_dir = '{:s}/reinit_{:d}_{:d}'.format(out_dir, int(reinit_err_thresh), reinit_frame_skip)
        else:
            in_arch_path = '{:s}/reinit_{:4.2f}_{:d}'.format(in_arch_path, reinit_err_thresh, reinit_frame_skip)
            tracking_root_dir = '{:s}/reinit_{:4.2f}_{:d}'.format(tracking_root_dir, reinit_err_thresh,
                                                                  reinit_frame_skip)
            out_dir = '{:s}/reinit_{:4.2f}_{:d}'.format(out_dir, reinit_err_thresh, reinit_frame_skip)

    out_dir = '{:s}/{:s}'.format(out_dir, actor)

    print 'tracking_root_dir: ', tracking_root_dir
    print 'in_arch_path: ', in_arch_path
    print 'out_dir: ', out_dir

    mr_success_rates = []
    mr_cmb_tracking_err = []
    mr_cmb_tracking_err_list = []
    mr_cmb_failure_count = []
    mr_frames_per_failure = []
    mr_cmb_avg_err = []

    for run_id in xrange(n_runs):
        success_rates = np.zeros((n_err_thr, n_seq + 2), dtype=np.float64)
        success_rates[:, 0] = err_thresholds

        cmb_tracking_err = []
        cmb_tracking_err_list = []

        cmb_failure_count = np.zeros((1, n_seq + 2), dtype=np.float64)
        frames_per_failure = np.zeros((1, n_seq + 2), dtype=np.float64)
        cmb_avg_err = np.zeros((1, n_seq + 2), dtype=np.float64)

        total_frame_count = 0

        for j in xrange(n_seq):
            seq_id = seq_ids[j]
            seq_name = sequences[seq_id]
            if actor == 'Synthetic':
                seq_name = getSyntheticSeqName(seq_name, syn_ssm, syn_ssm_sigma_id, syn_ilm,
                                               syn_am_sigma_id, syn_frame_id, syn_add_noise,
                                               syn_noise_mean, syn_noise_sigma)

            if use_reinit_gt:
                if use_opt_gt:
                    gt_path = '{:s}/{:s}_{:s}.bin'.format(gt_dir, seq_name, opt_gt_ssm)
                else:
                    gt_path = '{:s}/{:s}.bin'.format(gt_dir, seq_name)
            else:
                if use_opt_gt:
                    gt_path = '{:s}/{:s}_{:s}.txt'.format(gt_dir, seq_name, opt_gt_ssm)
                else:
                    gt_path = '{:s}/{:s}.txt'.format(gt_dir, seq_name)
            if use_arch:
                tracking_res_path = '{:s}/{:s}/{:s}/{:s}_{:s}_{:s}_{:d}'.format(
                    in_arch_path, actor, seq_name, mtf_sm, mtf_am, mtf_ssm, iiw)
            else:
                if using_pf:
                    tracking_res_path = '{:s}/pf/{:s}'.format(
                        tracking_root_dir, seq_name)
                else:
                    tracking_res_path = '{:s}/{:s}/{:s}/{:s}_{:s}_{:s}_{:d}'.format(
                        tracking_root_dir, actor, seq_name, mtf_sm, mtf_am, mtf_ssm, iiw)
            if enable_subseq:
                start_ids = subseq_start_ids[seq_id, :]
                # print 'seq_name: {:s}'.format(seq_name)

            tracking_res_path = '{:s}_run_{:d}.txt'.format(tracking_res_path, run_id + 1)
            if not use_arch and not os.path.isfile(tracking_res_path):
                print 'tracking result file not found for: {:s} in: \n {:s} '.format(seq_name, tracking_res_path)
                sys.exit(0)
            tracking_err, failure_count = getTrackingErrors(tracking_res_path, gt_path, arch_fid,
                                                            reinit_on_failure, reinit_frame_skip, use_reinit_gt,
                                                            start_ids, err_type, overflow_err, show_jaccard_img)
            if tracking_err is None:
                raise StandardError('Tracking error could not be computed for sequence {:d}: {:s}'.format(
                    j, seq_name))
            total_frame_count += len(tracking_err)
            cmb_tracking_err.append(tracking_err)
            if reinit_at_each_frame:
                avg_err = float(sum(tracking_err)) / float(len(tracking_err))
                cmb_avg_err[0, j + 1] = avg_err
                cmb_tracking_err_list.extend(tracking_err)
            if reinit_on_failure:
                cmb_tracking_err_list.extend(tracking_err)
                cmb_failure_count[0, j + 1] = failure_count
                frames_per_failure[0, j + 1] = float(len(tracking_err)) / float(failure_count + 1)
                if len(tracking_err) == 0:
                    print 'Tracker failed in all frames so setting the average error to the threshold'
                    avg_err = reinit_err_thresh
                else:
                    avg_err = float(sum(tracking_err)) / float(len(tracking_err))
                cmb_avg_err[0, j + 1] = avg_err
        print 'total_frame_count: ', total_frame_count
        if reinit_at_each_frame:
            cmb_avg_err[0, n_seq + 1] = float(sum(cmb_tracking_err_list)) / float(len(cmb_tracking_err_list))
        if reinit_on_failure:
            total_failure_count = np.sum(cmb_failure_count)
            if reinit_on_failure:
                print 'total_failure_count: {:d}'.format(int(total_failure_count))
            print 'total_error_count: ', len(cmb_tracking_err_list)
            cmb_failure_count[0, n_seq + 1] = total_failure_count
            cmb_avg_err[0, n_seq + 1] = float(sum(cmb_tracking_err_list)) / float(len(cmb_tracking_err_list))
            frames_per_failure[0, n_seq + 1] = float(total_frame_count) / float(total_failure_count + 1)

        # write tracking errors
        if write_err:
            if reinit_at_each_frame:
                actor_err_out_dir = '{:s}/reinit'.format(err_out_dir)
            elif reinit_on_failure:
                if reinit_err_thresh == int(reinit_err_thresh):
                    actor_err_out_dir = '{:s}/reinit_{:d}_{:d}'.format(err_out_dir, int(reinit_err_thresh),
                                                                       reinit_frame_skip)
                else:
                    actor_err_out_dir = '{:s}/reinit_{:4.2f}_{:d}'.format(err_out_dir, reinit_err_thresh, reinit_frame_skip)
            else:
                actor_err_out_dir = err_out_dir
            actor_err_out_dir = '{:s}/{:s}'.format(actor_err_out_dir, actor)
            if not os.path.exists(actor_err_out_dir):
                print 'Tracking error directory: {:s} does not exist. Creating it...'.format(actor_err_out_dir)
                os.makedirs(actor_err_out_dir)
            err_out_path = '{:s}/err_{:s}_{:s}_{:s}_{:d}'.format(
                actor_err_out_dir, mtf_sm, mtf_am, mtf_ssm, iiw)
            if use_opt_gt:
                err_out_path = '{:s}_{:s}'.format(err_out_path, opt_gt_ssm)
            if enable_subseq:
                err_out_path = '{:s}_subseq_{:d}'.format(err_out_path, n_subseq)
            if err_type:
                err_out_path = '{:s}_{:s}'.format(err_out_path, err_postfix)
            err_out_path = '{:s}_{:d}.txt'.format(err_out_path, run_id + 1)

            print 'Writing tracking errors to: {:s}'.format(err_out_path)
            # np.savetxt(err_out_path, np.array([np.array(err) for err in cmb_tracking_err]), fmt='%15.9f', delimiter='\t')
            cmb_tracking_err_arr = list(it.izip_longest(*cmb_tracking_err, fillvalue=-1))
            # print 'cmb_tracking_err_arr: ', cmb_tracking_err_arr
            np.savetxt(err_out_path, np.array([k for k in cmb_tracking_err_arr]), fmt='%15.9f', delimiter='\t')

        # compute success rates for different thresholds
        for i in xrange(n_err_thr):
            success_frame_count = 0
            err_thr = err_thresholds[i]
            for j in xrange(n_seq):
                seq_success_count = sum(err <= err_thr for err in cmb_tracking_err[j])
                success_frame_count += seq_success_count
                if len(cmb_tracking_err[j]) == 0:
                    success_rates[i, j + 1] = 0
                else:
                    success_rates[i, j + 1] = float(seq_success_count) / float(len(cmb_tracking_err[j]))
            success_rates[i, n_seq + 1] = float(success_frame_count) / float(total_frame_count)
            print 'err_thr: {:15.9f} sfc: {:6d} tfc: {:6d} sr: {:15.9f}'.format(
                err_thr, success_frame_count, total_frame_count, success_rates[i, n_seq + 1])
        mr_success_rates.append(success_rates)
        mr_cmb_tracking_err.append(cmb_tracking_err)
        mr_cmb_failure_count.append(cmb_failure_count)
        mr_frames_per_failure.append(frames_per_failure)
        mr_cmb_avg_err.append(cmb_avg_err)

    success_rates = np.mean(np.dstack(mr_success_rates), axis=2)
    cmb_failure_count = np.mean(np.asarray(mr_cmb_failure_count), axis=0)
    frames_per_failure = np.mean(np.asarray(mr_frames_per_failure), axis=0)
    cmb_avg_err = np.mean(np.asarray(mr_cmb_avg_err), axis=0)

    # write success rates (and other results if available)
    if not os.path.exists(out_dir):
        print 'Output directory: {:s} does not exist. Creating it...'.format(out_dir)
        os.makedirs(out_dir)

    out_path = '{:s}/sr'.format(out_dir)
    if actor == 'Synthetic':
        syn_out_suffix = getSyntheticSeqSuffix(syn_ssm, syn_ssm_sigma_id, syn_ilm, syn_am_sigma_id,
                                               syn_add_noise, syn_noise_mean, syn_noise_sigma)
        out_path = '{:s}_{:s}'.format(out_path, syn_out_suffix)

    if overriding_seq_id >= 0:
        out_path = '{:s}_{:s}'.format(out_path, sequences[overriding_seq_id])

    out_path = '{:s}_{:s}_{:s}_{:s}_{:d}'.format(
        out_path, mtf_sm, mtf_am, mtf_ssm, iiw)
    if use_opt_gt:
        out_path = '{:s}_{:s}'.format(out_path, opt_gt_ssm)
    if enable_subseq:
        out_path = '{:s}_subseq_{:d}'.format(out_path, n_subseq)
    if err_type:
        out_path = '{:s}_{:s}'.format(out_path, err_postfix)

    out_path = '{:s}_{:d}_runs'.format(out_path, n_runs)

    if write_to_bin:
        out_path = '{:s}.bin'.format(out_path)
    else:
        out_path = '{:s}.txt'.format(out_path)
    print 'Saving success rate data to {:s}'.format(out_path)
    if reinit_at_each_frame:
        out_data = np.vstack((success_rates, cmb_avg_err))
    elif reinit_on_failure:
        out_data = np.vstack((success_rates, cmb_avg_err, cmb_failure_count, frames_per_failure))
    else:
        out_data = success_rates
    if write_to_bin:
        out_file = open(out_path, 'wb')
        np.array(out_data.shape, dtype=np.uint32).tofile(out_file)
        out_data.astype(np.float64).tofile(out_file)
        out_file.close()
    else:
        np.savetxt(out_path, out_data, fmt='%15.9f', delimiter='\t')

    if use_arch:
        arch_fid.close()
