from datetime import datetime
import os
import glob
import json
import pandas as pd

import paramparse

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


class Params:

    def __init__(self):
        self.cfg_ext = 'cfg'
        self.cfg_root = 'cfg/concat'

        self.list_path_id = 19

        # self.cfg = 'tp_fp'
        # self.cfg = 'roc_auc'
        # self.cfg = 'rec_prec'
        self.cfg = 'fpr_tpr'

        self.in_dir = 'cmd/concat'
        self.list_ext = 'list'

        """1-based ID of files within "in_dir" in lexical order"""

        self.list_path = 'inv-swi-inc-nms'
        # self.list_path = 'inv_all'
        # self.list_path = 'inv_all_seg'

        # self.list_path = 'fwd_all_seg'
        # self.list_path = 'fwd_all'

        self.in_dirs_root = 'log/seg'

        self.csv_mode = 0
        self.out_dir = 'log'
        self.out_path = 'list_out'

        # self.class_name = ''
        self.class_name = 'ipsc'

        self.json_name = 'eval_result_dict.json'
        self.json_metrics = ['FNR_DET', 'FN_DET', 'FPR_DET', 'FP_DET']
        """
        axis=0 --> one metric in each row / one metric for all models concatenated horizontally
        axis=1 --> one metric in each column / all metrics for each model concatenated horizontally
        """
        self.axis = 0

        self.clipboard = 0

        self.csv_metrics = [
            # 'tp_fp_cls',

            # 'tp_fp_uex',
            # 'tp_fp_uex_fn',
            # 'tp_fp_ex',
            # 'tp_fp_ex_fn',

            # 'roc_auc_cls',

            # 'rec_prec',

        ]
        self.sep = '\t'


def main():
    params = Params()
    paramparse.process(params)

    in_dirs = []
    if params.clipboard:
        try:
            in_txt = Tk().clipboard_get()  # type: str
        except BaseException as e:
            print('Tk().clipboard_get() failed: {}'.format(e))
        else:
            in_dirs = [k.strip() for k in in_txt.split('\n') if k.strip() and not k.startswith('#')]

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_name = time_stamp

    if params.list_path_id > 0:
        list_files = glob.glob(os.path.join(params.in_dir, f'**/*.{params.list_ext}'), recursive=True)
        list_files = sorted(list_files)
        list_path = list_files[params.list_path_id - 1]
    else:
        list_path = linux_path(params.in_dir, params.list_path)
        if params.list_ext:
            list_path = f'{list_path}.{params.list_ext}'

    if not in_dirs or not all(os.path.isdir(in_dir)
                              # and os.path.isfile(linux_path(in_dir, params.json_name))
                              for in_dir in in_dirs):
        print(f'reading in_dirs from {list_path}')
        in_dirs = open(list_path, 'r').read().splitlines()
        in_dirs = [k.strip() for k in in_dirs if k.strip() and not k.startswith('#')]

        in_name = os.path.splitext(os.path.basename(list_path))[0]
        out_name = f'{out_name}_{in_name}'

        if params.cfg:
            out_name = f'{out_name}_{params.cfg}'

    else:
        in_name = ''

    models = []
    for in_dir_id, in_dir in enumerate(in_dirs):
        try:
            model, in_dir = in_dir.split('\t')
        except ValueError:
            model = os.path.basename(in_dir)
        else:
            in_dirs[in_dir_id] = in_dir

        models.append(model)
    # if params.axis == 0:
    #     header = 'metric' + '\t' + '\t'.join(models)
    # else:
    #     header = 'model' + '\t' + '\t'.join(params.json_metrics)

    if params.in_dirs_root:
        in_dirs = [linux_path(params.in_dirs_root, k) for k in in_dirs]

    metrics_dict = {
        model: {} for model in models
    }

    for in_dir, model in zip(in_dirs, models):
        if params.csv_mode:
            csv_names = [f'{metric}.csv' for metric in params.csv_metrics]
            if params.class_name:
                csv_names = [f'{params.class_name}-{csv_name}' for csv_name in csv_names]

            csv_paths = [linux_path(in_dir, csv_name) for csv_name in csv_names]
            for csv_id, csv_path in enumerate(csv_paths):
                """look for recursive alternatives if needed"""
                if not os.path.exists(csv_path):
                    print(f'\ncsv_path does not exist: {csv_path}')
                    matching_files = glob.glob(os.path.join(in_dir, f'**/{csv_names[csv_id]}'), recursive=True)
                    if not matching_files:
                        raise AssertionError('no alternative matching files found')
                    if len(matching_files) > 1:
                        raise AssertionError('multiple alternative matching files found')
                    print(f'found alternative: {matching_files[0]}\n')
                    csv_paths[csv_id] = matching_files[0]

            metric_dfs = [pd.read_csv(csv_path, sep='\t') for csv_path in csv_paths]

            metrics_dict[model] = metric_dfs
        else:
            json_path = os.path.join(in_dir, params.json_name)
            with open(json_path, 'r') as fid:
                json_data = json.load(fid)

            if params.class_name:
                json_data = json_data[params.class_name]

            for metric in params.json_metrics:
                metric_val = json_data[metric]

                metrics_dict[model][metric] = metric_val

    if not params.csv_mode:
        out_name = f'{out_name}_json'

    out_dir = linux_path(params.out_dir, out_name)
    os.makedirs(out_dir, exist_ok=1)

    if params.csv_mode:
        # axis_0_txt = ''
        all_out_lines = []
        for metric_id, metric in enumerate(params.csv_metrics):
            # axis_0_txt += f'\n{metric}\n'
            all_models_dfs = [metrics_dict[model][metric_id] for model in models]
            n_cols = [all_models_df.shape[1] for all_models_df in all_models_dfs]
            header_list = [model + '\t' * (n_cols - 1) for n_cols, model in zip(n_cols, models)]
            header2 = '\t'.join(header_list)

            all_models_dfs_cc = pd.concat(all_models_dfs, axis=1)

            df_txt = all_models_dfs_cc.to_csv(sep='\t', index=False, line_terminator='\n')

            header1 = f'{metric}' + '\t' * (all_models_dfs_cc.shape[1] - 1)

            out_txt = header1 + '\n' + header2 + '\n' + df_txt + f'\n'

            open(linux_path(out_dir, f'{metric}.csv'), 'w', newline='').write(out_txt)

            out_lines = out_txt.splitlines()
            all_out_lines.append(out_lines)

        joined_out_lines = []
        for all_curr_lines in zip(*all_out_lines):
            out_line = '\t\t'.join(all_curr_lines)
            joined_out_lines.append(out_line)

        axis_0_txt = '\n'.join(joined_out_lines)

        open(linux_path(out_dir, f'concat_axis_0.csv'), 'w', newline='\n').write(axis_0_txt)

        all_out_lines = []
        for model in models:
            all_metrics_dfs = metrics_dict[model]
            n_cols = [all_metrics_df.shape[1] for all_metrics_df in all_metrics_dfs]
            header_list = [metric + '\t' * (n_cols - 1) for n_cols, metric in zip(n_cols, params.csv_metrics)]
            header2 = '\t'.join(header_list)

            all_metrics_dfs_cc = pd.concat(all_metrics_dfs, axis=1)

            df_txt = all_metrics_dfs_cc.to_csv(sep='\t', index=False, line_terminator='\n')

            header1 = f'{model}' + '\t' * (all_metrics_dfs_cc.shape[1] - 1)

            out_txt = header1 + '\n' + header2 + '\n' + df_txt + f'\n'

            open(linux_path(out_dir, f'{model}.csv'), 'w', newline='').write(out_txt)

            out_lines = out_txt.splitlines()
            all_out_lines.append(out_lines)

        axis_1_txt = '\n'.join('\t\t'.join(all_curr_lines) for all_curr_lines in zip(*all_out_lines))

        open(linux_path(out_dir, f'concat_axis_1.csv'), 'w', newline='\n').write(axis_1_txt)

        if params.axis == 0:
            out_txt = axis_0_txt
        else:
            out_txt = axis_1_txt

        try:
            import pyperclip

            pyperclip.copy(out_txt)
            spam = pyperclip.paste()
        except BaseException as e:
            print('Copying to clipboard failed: {}'.format(e))
    else:
        metrics_df = pd.DataFrame.from_dict(metrics_dict)
        metrics_df_t = metrics_df.transpose()

        if params.axis == 0:
            metrics_df.to_clipboard(sep='\t', index_label='metric')
        else:
            metrics_df_t.to_clipboard(sep='\t', index_label='model')

        metrics_header = '/'.join(params.json_metrics)

        with open(linux_path(out_dir, 'metrics_df.csv'), 'w', newline='') as fid:
            if in_name:
                fid.write(in_name + '\n')
            fid.write(metrics_header + '\n')
            metrics_df.to_csv(fid, sep='\t', index_label='metric')

        with open(linux_path(out_dir, 'metrics_df_t.csv'), 'w', newline='') as fid:
            fid.write(metrics_header + '\n')
            if in_name:
                fid.write(in_name + '\n')
            metrics_df.to_csv(fid, sep='\t', index_label='metric')
            metrics_df_t.to_csv(fid, sep='\t', index_label='model')


if __name__ == '__main__':
    main()
