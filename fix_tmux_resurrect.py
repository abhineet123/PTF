import os


def main():
    tr_path = os.path.expanduser('~/.tmux/resurrect')
    assert os.path.isdir(tr_path), f"invalid tr_path: {tr_path}"

    last_resurrect_ln_path = os.path.join(tr_path, 'last')
    if os.path.exists(last_resurrect_ln_path):
        print(f'removing {last_resurrect_ln_path}')
        # os.remove(last_resurrect_ln_path)

    resurrect_files = [os.path.join(tr_path, k) for k in os.listdir(tr_path)
                       if k.startswith('tmux_resurrect_') and k.endswith('.txt')]

    assert resurrect_files, "no resurrect_files found"

    resurrect_files_size = [(k, os.path.getsize(k)) for k in resurrect_files]
    resurrect_files_size = sorted(resurrect_files_size, key=lambda x: x[1])

    largest_size = float(resurrect_files_size[-1][1])
    resurrect_files_mtime = []
    for k, size_ in resurrect_files_size:
        size_ratio = float(size_) / largest_size
        if size_ratio < 0.5:
            print(f'removing {k}')
            # os.remove(k)
        else:
            resurrect_files_mtime.append((k, os.path.getmtime(k)))

    assert resurrect_files_mtime, "no valid resurrect_files left"

    resurrect_files_mtime = sorted(resurrect_files_mtime, key=lambda x: x[1])
    newest_file = resurrect_files_mtime[-1][0]
    print(f'newest_file: {newest_file}')
    # os.symlink(newest_file[0], last_resurrect_ln_path)


if __name__ == '__main__':
    main()
