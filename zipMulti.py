import os
import sys
import shutil

from datetime import datetime

from Misc import processArguments

if __name__ == '__main__':
    params = {
        'dir_names': [],
        'exclusions': [],
        'exclude_ext': [],
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
    exclude_ext = params['exclude_ext']
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

    zip_dir = os.path.dirname(zip_path)
    zip_fname = os.path.basename(zip_path)
    zip_fname_noext, zip_fname_ext = os.path.splitext(zip_fname)
    if zip_fname_ext == '.zip':
        out_name = zip_fname_noext
        if postfix:
            out_name = '{}_{}'.format(out_name, postfix)

        out_name = out_name.replace('.', '_')
        if add_time_stamp:
            time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
            out_name = '{}_{}'.format(out_name, time_stamp)
        out_name = '{}.zip'.format(out_name)

        out_path = os.path.join(zip_dir, out_name)

        shutil.copy(zip_path, out_path)
    else:
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

        if relative:
            zip_cmd = 'cd {} && zip {} {} {}'.format(zip_root_path, switches, out_name, zip_file)
            out_path = os.path.join(zip_root_path, out_name)
            exclude_root = zip_file
        else:
            zip_cmd = 'zip {:s} {:s} {:s}'.format(switches, out_name, zip_path)
            out_path = out_name
            exclude_root = zip_path

        if exclusions:
            print('Excluding files / folders: {}'.format(exclusions))
            switches2 = ''
            for exclusion in exclusions:
                if os.path.isabs(exclusion):
                    exclusion = os.path.relpath(exclusion, exclude_root)
                if exclude_root not in exclusion:
                    exclusion = os.path.join(exclude_root, exclusion)
                switches2 += ' -x "{}"'.format(exclusion)
            zip_cmd = '{:s} {:s}'.format(zip_cmd, switches2)

        if exclude_ext:
            print('Excluding files with extensions: {}'.format(exclude_ext))
            switches2 = ''
            for _ext in exclude_ext:
                switches2 += ' -x "*.{}"'.format(_ext)
                switches2 += ' -x "*.{}.*"'.format(_ext)
            zip_cmd = '{:s} {:s}'.format(zip_cmd, switches2)

        print(zip_cmd)
        os.system(zip_cmd)

    assert os.path.exists(out_path), "zipping failed: {}".format(out_path)

    # os.system('unzip -l {}'.format(out_path))

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
