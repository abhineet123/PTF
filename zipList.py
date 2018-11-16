import os, sys
from datetime import datetime

from Misc import processArguments, sortKey

if __name__ == '__main__':
    params = {
        'list_file': '',
        'file_name': '',
        'root_dir': '',
        'out_name': '',
        'scp_dst': '',
        'out_postfix': '',
    }
    processArguments(sys.argv[1:], params)
    list_file = params['list_file']
    root_dir = params['root_dir']
    file_name = params['file_name']
    out_name = params['out_name']
    out_postfix = params['out_postfix']
    scp_dst = params['scp_dst']

    if list_file:
        if os.path.isdir(list_file):
            zip_paths = [os.path.join(list_file, name) for name in os.listdir(list_file) if
                         os.path.isdir(os.path.join(list_file, name))]
            zip_paths.sort(key=sortKey)
        else:
            zip_paths = [x.strip() for x in open(list_file).readlines() if x.strip()]
            if root_dir:
                zip_paths = [os.path.join(root_dir, name) for name in zip_paths]
    else:
        zip_paths = [file_name]

    # zip_paths = [os.path.abspath(k) for k in zip_paths]

    if not out_name:
        _root_dir = os.path.basename(os.path.dirname(os.path.abspath(zip_paths[0])))
        list_fname_no_ext = os.path.splitext(os.path.basename(list_file))[0]
        out_name = '{}_{}'.format(_root_dir, list_fname_no_ext)
    else:
        out_name = os.path.splitext(out_name)[0]

    if out_postfix:
        out_name = '{}_{}'.format(out_name, out_postfix)

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_name = '{}_{}.zip'.format(out_name, time_stamp)

    zip_cmd = 'zip -r {:s}'.format(out_name)
    for zip_path in zip_paths:
        zip_cmd = '{:s} {:s}'.format(zip_cmd, zip_path)

    print('\nrunning: {}\n'.format(zip_cmd))
    os.system(zip_cmd)

    if scp_dst:
        scp_cmd = 'scp {} {}:~/'.format(out_name, scp_dst)
        print('\nrunning: {}\n'.format(scp_cmd))
        os.system(scp_cmd)
        rm_cmd = 'rm {}'.format(out_name)
        print('\nrunning: {}\n'.format(rm_cmd))
        os.system(rm_cmd)
    else:
        mv_cmd = 'mv {:s} ~'.format(out_name)
        print('\nrunning: {}\n'.format(mv_cmd))
        os.system(mv_cmd)

    print('out_name:\n {}'.format(out_name))

