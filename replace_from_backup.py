import os
import shutil
import tqdm
import paramparse


class Params:
    def __init__(self):
        self.src_dir = ''
        self.dst_dir = ''
        self.recursive = 1
        self.remove_suffix = 1
        self.allow_missing = 1
        self.suffix_sep = '__'


def main():
    params = Params()
    paramparse.process(params)

    src_dir = params.src_dir
    dst_dir = params.dst_dir
    recursive = params.recursive
    remove_suffix = params.remove_suffix
    allow_missing = params.allow_missing

    print(f'src_dir: {src_dir}')
    print(f'dst_dir: {dst_dir}')

    if recursive:
        print('Searching recursively')
        src_file_gen = [[os.path.join(dirpath, f) for f in filenames if f != 'Thumbs.db']
                        for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)
                        if dirpath != os.getcwd()]
        src_file_paths = [item for sublist in src_file_gen for item in sublist]

        dst_file_gen = [[os.path.join(dirpath, f) for f in filenames if f != 'Thumbs.db']
                        for (dirpath, dirnames, filenames) in os.walk(dst_dir, followlinks=True)
                        if dirpath != os.getcwd()]
        dst_file_paths = [item for sublist in dst_file_gen for item in sublist]
    else:
        src_file_paths = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if
                          os.path.isfile(os.path.join(src_dir, f)) and f != 'Thumbs.db']

        dst_file_paths = [os.path.join(dst_dir, f) for f in os.listdir(dst_dir) if
                          os.path.isfile(os.path.join(dst_dir, f)) and f != 'Thumbs.db']

    n_src_files = len(src_file_paths)
    n_dst_files = len(dst_file_paths)

    print(f'n_src_files: {n_src_files}')
    print(f'n_dst_files: {n_dst_files}')

    assert n_src_files > 0, "no src_files found"
    assert allow_missing or n_dst_files >= n_src_files, "insufficient dst_files found"

    src_file_names = [os.path.basename(src_file_path) for src_file_path in src_file_paths]
    dst_file_names = [os.path.basename(dst_file_path) for dst_file_path in dst_file_paths]

    if remove_suffix:
        src_file_names = [src_file_name.split(params.suffix_sep)[0] for src_file_name in src_file_names]
        dst_file_names = [dst_file_name.split(params.suffix_sep)[0] for dst_file_name in dst_file_names]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'log')
    os.makedirs(log_dir, exist_ok=1)

    from datetime import datetime

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    log_path = os.path.join(log_dir, f'rfb_{time_stamp}.txt')

    print(f'log_path: {log_path}')

    pbar = tqdm.tqdm(src_file_names, ncols=100)

    n_missing = 0

    for src_file_id, src_file_name in enumerate(pbar):

        src_file_path = src_file_paths[src_file_id]

        try:
            dst_file_id = dst_file_names.index(src_file_name)
        except ValueError:
            msg = f'no matching dst file found for {src_file_name} : {src_file_path}'
            if not allow_missing:
                raise AssertionError(msg)
            # print(msg)
            n_missing += 1

            with open(log_path, 'a') as log_fid:
                log_fid.write(src_file_path + '\n')

            continue

        dst_file_path = dst_file_paths[dst_file_id]

        msg = f'{src_file_path}\t{dst_file_path}'

        # print(msg)
        with open(log_path, 'a') as log_fid:
            log_fid.write(msg + '\n')

        shutil.copy(src_file_path, dst_file_path)

        pbar.set_description(f'n_missing: {n_missing}')


if __name__ == '__main__':
    main()
