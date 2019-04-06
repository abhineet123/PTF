import os, sys
from datetime import datetime

from Misc import processArguments

if __name__ == '__main__':
    params = {
        'dir_names': [],
        'out_name': '',
        'postfix': '',
        'scp_dst': '',
        'switches': '-r',
        'relative': 0,
    }
    processArguments(sys.argv[1:], params)
    _dir_names = params['dir_names']
    out_name = params['out_name']
    postfix = params['postfix']
    switches = params['switches']
    scp_dst = params['scp_dst']
    relative = params['relative']

    print('dir_names: ', _dir_names)

    if len(_dir_names) == 1:
        dir_names = _dir_names[0].split(os.sep)
        if _dir_names.startswith(os.sep):
            del dir_names[0]
            dir_names[0] = os.sep + dir_names[0]
    else:
        dir_names = _dir_names

    print('dir_names: ', dir_names)

    zip_path = ''
    for _dir in dir_names:
        zip_path = os.path.join(zip_path, _dir) if zip_path else _dir

    print('zip_path: ', zip_path)

    out_start_id = 0
    if out_name:
        try:
            out_start_id = int(out_name)
        except ValueError:
            pass
        else:
            out_name = ''

    if not out_name:
        for _dir in dir_names[out_start_id:]:
            out_name = '{}_{}'.format(out_name, _dir) if out_name else _dir
    else:
        out_name = os.path.splitext(out_name)[0]

    if postfix:
        out_name = '{}_{}'.format(out_name, postfix)

    out_name = out_name.replace('.', '_')
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_name = '{}_{}.zip'.format(out_name, time_stamp)

    if os.path.isdir(zip_path):
        zip_root_path = zip_path
        zip_file = '*'
    elif os.path.isfile(zip_path):
        zip_root_path = os.path.dirname(zip_path)
        zip_file = os.path.basename(zip_path)
    else:
        raise IOError('zip_path is neither a folder nor a file')

    if relative:
        zip_cmd = 'cd {} && zip {} {} {}'.format(zip_root_path, switches, out_name, zip_file)
        out_path = os.path.join(zip_root_path, out_name)
    else:
        zip_cmd = 'zip {:s} {:s}'.format(switches, out_name)
        zip_cmd = '{:s} {:s}'.format(zip_cmd, zip_path)
        out_path = out_name

    os.system(zip_cmd)
    os.system('unzip -l {}'.format(out_path))

    if scp_dst:
        scp_cmd = 'scp {} {}:~/'.format(out_path, scp_dst)
        print('\nrunning: {}\n'.format(scp_cmd))
        os.system(scp_cmd)
        rm_cmd = 'rm {}'.format(out_path)
        print('\nrunning: {}\n'.format(rm_cmd))
        os.system(rm_cmd)
    else:
        mv_cmd = 'mv {:s} ~'.format(out_path)
        print('\nrunning: {}\n'.format(mv_cmd))
        os.system(mv_cmd)

    print('out_name:\n {}'.format(out_name))

    # import pyperclip
    # try:
    #     pyperclip.copy(out_name)
    #     spam = pyperclip.paste()
    # except pyperclip.PyperclipException as e:
    #     print('Copying to clipboard failed: {}'.format(e))
