import os, sys
from Misc import processArguments, sortKey
from pprint import pprint
if __name__ == '__main__':
    params = {
        'list_file': '',
        'template_file': '',
        'template_id': 1,
        'file_name': '',
        'root_dir': '',
        'dir_pattern': [],
        'out_file': 'consolidated_results.txt',
        'scp_dst': '',
        'out_postfix': '',
    }
    print(sys.argv)

    processArguments(sys.argv[1:], params)
    list_file = params['list_file']
    template_file = params['template_file']
    template_id = params['template_id']
    root_dir = params['root_dir']
    dir_pattern = params['dir_pattern']
    file_name = params['file_name']
    out_file = params['out_file']
    out_postfix = params['out_postfix']
    scp_dst = params['scp_dst']

    template_lines = (
        'pix_acc\t mean_acc\t mean_IU\t fw_IU',
        'mean_acc_ice\t mean_acc_ice_1\t mean_acc_ice_2',
        'mean_IU_ice\t mean_IU_ice_1\t mean_IU_ice_2'
    )
    template_line = template_lines[template_id]
    if template_file:
        template_line = open(template_file, 'r').readline().strip()

    results_exts = ['.txt']

    if list_file:
        if os.path.isdir(list_file):
            print('Looking for results files in {}'.format(list_file))
            results_paths_gen = [[os.path.join(dirpath, f) for f in filenames if
                             os.path.splitext(f.lower())[1] in results_exts]
                            for (dirpath, dirnames, filenames) in os.walk(list_file, followlinks=True)]
            results_paths = [item for sublist in results_paths_gen for item in sublist]

            if dir_pattern:
                print('Restricting search to folders containing:{}'.format(dir_pattern))
                results_paths = [x for x in results_paths if all([k in x for k in dir_pattern])]

            results_paths.sort(key=lambda x: sortKey(x, only_basename=0), reverse=True)
            unique_dirs = set()
            unique_paths = []
            for _path in results_paths:
                _dir = os.path.dirname(_path)
                if _dir in unique_dirs:
                    print('Ignoring outdated results file: {}'.format(_path))
                    continue
                unique_dirs.add(_dir)
                unique_paths.append(_path)
            results_paths = unique_paths
            results_paths.sort(key=lambda x: sortKey(x, only_basename=0), reverse=False)
        else:
            results_paths = [x.strip() for x in open(list_file).readlines() if x.strip()]
            if root_dir:
                results_paths = [os.path.join(root_dir, name) for name in results_paths]
    else:
        results_paths = [file_name]

    print('Searching for template line:\n{}'.format(template_line))
    print('Searching in {} files:'.format(len(results_paths)))
    pprint(results_paths)


    if not os.path.isfile(out_file):
        open(out_file, 'w').close()

    open(out_file, 'a').write('file\t{}\n'.format(template_line))

    result_line = result_vals = None
    for results_path in results_paths:
        results_lines = open(results_path, 'r').readlines()
        template_line_found = False
        for _line in results_lines:
            if template_line_found:
                result_line = _line
                try:
                    result_vals = [float(x) for x in _line.strip().split(sep='\t')]
                except ValueError as e:
                    raise IOError('Invalid results line: {} '.format(_line))
                break
            if _line.strip() == template_line:
                template_line_found = True
                continue
        if not template_line_found:
            print('Template line not found in {}'.format(results_path))
            continue
        open(out_file, 'a').write('{}\t{}\n'.format(results_path, result_line.strip()))



