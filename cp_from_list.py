import os, sys, glob, re

from Misc import processArguments, sortKey

if __name__ == '__main__':
    params = {
        'list_file': '',
        'file_name': '',
        'root_dir': '.',
        'dst_path': '',
    }
    processArguments(sys.argv[1:], params)
    list_file = params['list_file']
    root_dir = params['root_dir']
    file_name = params['file_name']
    dst_path = params['dst_path']

    if not dst_path or not os.path.isdir(dst_path):
        raise IOError('dst_path is invalid: {}'.format(dst_path))

    if list_file:
        if os.path.isdir(list_file):
            src_paths = [os.path.join(list_file, name) for name in os.listdir(list_file) if
                         os.path.isdir(os.path.join(list_file, name))]
            src_paths.sort(key=sortKey)
        else:
            src_paths = [x.strip() for x in open(list_file).readlines() if x.strip() and not x.startswith('#')]
            if root_dir:
                root_dir = os.path.abspath(root_dir)
                src_paths = [os.path.join(root_dir, name) for name in src_paths]
        out_file_path = '{}.out'.format(list_file)
    else:
        src_paths = [file_name]
        out_file_path = '{}.out'.format(file_name)

    n_src_paths = len(src_paths)
    for i, src_path in enumerate(src_paths):
        if src_path.startswith('#'):
            continue
        if not os.path.exists(src_path):
            print('src_path does not exist: {}'.format(src_path))
            continue
        # cp_cmd = 'rsync -r -ah --progress "{}" "{}"'.format(src_path, dst_path)
        cp_cmd = 'cp -r "{}" "{}"'.format(src_path, dst_path)
        print('\n{}/{} :: running: {}\n'.format(i+1, n_src_paths, cp_cmd))
        os.system(cp_cmd)
        src_path_full = os.path.abspath(src_path)
        with open(out_file_path, "a") as fid:
            fid.write(src_path_full + "\n")

