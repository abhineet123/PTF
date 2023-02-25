import os
from tqdm import tqdm

import paramparse

import pandas as pd


class Params:

    def __init__(self):
        self.dst_path = ''
        self.file_list = ''
        self.root_dir = ''


if __name__ == '__main__':
    params = Params()
    paramparse.process(params)

    file_list = params.file_list
    root_dir = params.root_dir
    dst_path = params.dst_path

    assert file_list, "file_list must be provided"

    file_list_name = os.path.splitext(os.path.basename(file_list))[0]

    if file_list:
        if os.path.isdir(file_list):
            src_paths = [os.path.join(file_list, name) for name in os.listdir(file_list) if
                         os.path.isdir(os.path.join(file_list, name))]
            src_paths.sort()
        else:
            src_paths = [x.strip() for x in open(file_list).readlines() if x.strip()
                         and not x.startswith('#')
                         and not x.startswith('@')]
            if root_dir:
                src_paths = [os.path.join(root_dir, name) for name in src_paths]



    n_src_paths = len(src_paths)

    all_dfs = []

    out_df_cols = None

    for i, src_path in enumerate(tqdm(src_paths)):
        if src_path.startswith('#'):
            continue

        src_name, src_ext = os.path.splitext(os.path.basename(src_path))

        if not src_ext:
            src_path = src_path + '.csv'

        if not os.path.exists(src_path):
            raise IOError('src_path does not exist: {}'.format(src_path))

        df = pd.read_csv(src_path)

        df_cols = tuple(df.columns)

        if out_df_cols is None:
            out_df_cols = df_cols
        else:
            assert out_df_cols == df_cols, "df_cols mismatch"

        all_dfs.append(df)

    if not dst_path:
        dst_path = f'{file_list_name}.csv'
    if root_dir:
        dst_path = os.path.join(root_dir, dst_path)
        
    print(f'saving combined csv to {dst_path}')
    combined_df = pd.concat(all_dfs, axis=0)
    combined_df.to_csv(dst_path, columns=out_df_cols, index=False)
