import os, sys
from datetime import datetime

from Misc import processArguments

if __name__ == '__main__':
    params = {
        'dir_names': [],
        'exclusions': [],
        'out_name': '',
        'postfix': '',
        'scp_dst': '',
        'switches': '-r',
        'relative': 0,
        'add_time_stamp': 1,
        'move_to_home': 1,
    }
    processArguments(sys.argv[1:], params)
    _dir_names = params['dir_names']
    exclusions = params['exclusions']
    out_name = params['out_name']
    postfix = params['postfix']
    switches = params['switches']
    scp_dst = params['scp_dst']
    relative = params['relative']
    add_time_stamp = params['add_time_stamp']
    move_to_home = params['move_to_home']

    print('dir_names: ', _dir_names)

    if len(_dir_names) == 1:
        dir_names = _dir_names[0].split(os.sep)
        if _dir_names[0].startswith(os.sep):
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

    if dir_names[0].startswith(os.sep):
        dir_names[0] = dir_names[0].replace(os.sep, '')

    if not out_name:
        for _dir in dir_names[out_start_id:]:
            out_name = '{}_{}'.format(out_name, _dir) if out_name else _dir
    else:
        out_name = os.path.splitext(out_name)[0]

    if postfix:
        out_name = '{}_{}'.format(out_name, postfix)

    out_name = out_name.replace('.', '_')
    if add_time_stamp:
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        out_name = '{}_{}'.format(out_name, time_stamp)
    out_name = '{}.zip'.format(out_name)

    if os.path.isdir(zip_path):
        zip_root_path = zip_path
        zip_file = '*'
    elif os.path.isfile(zip_path):
        zip_root_path = os.path.dirname(zip_path)
        zip_file = os.path.basename(zip_path)
    else:
        raise IOError('zip_path is neither a folder nor a file')

    switches2 = ''
    if exclusions:
        for exclusion in exclusions:
            rel_path = os.path.relpath(exclusion, zip_path)
            switches2 += ' --exclude {}'.format(rel_path)

    if relative:
        zip_cmd = 'cd {} && zip {} {} {}'.format(zip_root_path, switches, out_name, zip_file, switches2)
        out_path = os.path.join(zip_root_path, out_name)
    else:
        zip_cmd = 'zip {:s} {:s}'.format(switches, out_name)
        zip_cmd = '{:s} {:s} {:s}'.format(zip_cmd, zip_path, switches2)
        out_path = out_name

    print(zip_cmd)

    os.system(zip_cmd)

    assert os.path.exists(out_path), "zipping failed"

    os.system('unzip -l {}'.format(out_path))

    if scp_dst:
        scp_cmd = 'scp {} {}:~/'.format(out_path, scp_dst)
        print('\nrunning: {}\n'.format(scp_cmd))
        os.system(scp_cmd)
        rm_cmd = 'rm {}'.format(out_path)
        print('\nrunning: {}\n'.format(rm_cmd))
        os.system(rm_cmd)
    elif move_to_home:
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
