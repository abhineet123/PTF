import os
import sys
import glob
import shutil

from datetime import datetime

import paramparse


class Params:

    def __init__(self):
        self.cfg = ()
        self.add_time_stamp = 1
        self.dir_names = []
        self.exclude_ext = []
        self.exclusions = []
        self.include_ext = []
        self.inclusions = []
        self.move_to_home = 1
        self.out_name = ''
        self.postfix = ''
        self.relative = 0
        self.scp_dst = ''
        self.scp_port = ''
        self.switches = ''
        self.recursive = 1


if __name__ == '__main__':

    params = Params()
    paramparse.process(params)

    _dir_names = params.dir_names
    inclusions = params.inclusions
    exclusions = params.exclusions
    exclude_ext = params.exclude_ext
    include_ext = params.include_ext
    _out_name = params.out_name
    postfix = params.postfix
    switches = params.switches
    scp_dst = params.scp_dst
    scp_port = params.scp_port
    relative = params.relative
    add_time_stamp = params.add_time_stamp
    move_to_home = params.move_to_home
    recursive = params.recursive

    print('_dir_names: ', _dir_names)

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
        if _out_name:
            try:
                out_start_id = int(_out_name)
            except ValueError:
                pass
            else:
                out_name = ''

        if dir_names[0].startswith(os.sep):
            dir_names[0] = dir_names[0].replace(os.sep, '')

        # if not out_name:
        #     for _dir in dir_names[out_start_id:]:
        #         out_name = '{}_{}'.format(out_name, _dir) if out_name else _dir
        # else:
        #     out_name = os.path.splitext(out_name)[0]

        out_name = ''

        for _dir in dir_names[out_start_id:]:
            out_name = '{}_{}'.format(out_name, _dir) if out_name else _dir

        if postfix:
            out_name = '{}_{}'.format(out_name, postfix)

        out_name = out_name.replace('.', '_')
        out_name = out_name.replace('(', '_')
        out_name = out_name.replace(')', '_')
        if add_time_stamp:
            time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
            out_name = '{}_{}'.format(out_name, time_stamp)
        out_name = '{}.zip'.format(out_name)

        # if os.path.isdir(zip_path):
        #     zip_root_path = zip_path
        #     zip_file = '*'
        # elif os.path.isfile(zip_path):
        #     zip_root_path = os.path.dirname(zip_path)
        #     zip_file = os.path.basename(zip_path)
        # else:
        #     raise IOError('zip_path is neither a folder nor a file')

        zip_root_path = os.path.dirname(zip_path)
        zip_file = os.path.basename(zip_path)

        if _out_name:
            out_name = _out_name

        if recursive:
            switches += ' -r'

        if relative:
            zip_cmd = f'cd "{zip_root_path}" && zip {switches} "{out_name}" "{zip_file}"'
            out_path = os.path.join(zip_root_path, out_name)
            exclude_root = zip_file
        else:
            zip_cmd = f'zip {switches} "{out_name:s}" "{zip_path:s}"'
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

        if inclusions:
            switches2 = ''

            if inclusions[0] == '__pt__':
                print('Including only the last pytorch checkpoint')
                ckpt_files = sorted([file for file in os.listdir(zip_path) if file.endswith(".pt")])
                if len(ckpt_files) > 1:
                    excluded_ckpt_files = ckpt_files[:-1]
                    excluded_ckpt_names = [os.path.splitext(os.path.basename(k))[0] for k in excluded_ckpt_files]
                    for excluded_ckpt_name in excluded_ckpt_names:
                        switches2 += ' -i "{}.*"'.format(excluded_ckpt_name)

            elif inclusions[0] == '__tf__':
                print('Including only the last TF checkpoint')
                ckpt_files = sorted([file for file in os.listdir(zip_path) if file.endswith(".index")])
                if len(ckpt_files) > 1:
                    excluded_ckpt_files = ckpt_files[:-1]
                    excluded_ckpt_names = [os.path.splitext(os.path.basename(k))[0] for k in excluded_ckpt_files]
                    for excluded_ckpt_name in excluded_ckpt_names:
                        switches2 += ' -i "{}.*"'.format(excluded_ckpt_name)
            else:
                print('Including only files matching patterns: {}'.format(inclusions))
                for inclusion in inclusions:
                    if inclusion.startswith('__a__'):
                        inclusion = inclusion.replace('__a__', '')
                    else:
                        inclusion = f'*{inclusion}*'

                    switches2 += f' -i "{inclusion}"'

            zip_cmd = '{:s} {:s}'.format(zip_cmd, switches2)

        if include_ext:
            print('Including only files with extensions: {}'.format(include_ext))
            switches2 = ''
            for _ext in include_ext:
                switches2 += ' -i "*.{}"'.format(_ext)
            zip_cmd = '{:s} {:s}'.format(zip_cmd, switches2)
        elif exclude_ext:
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

    out_size = os.path.getsize(out_path) / 1000

    if scp_dst:
        scp_cmd = 'scp'
        if scp_port:
            scp_cmd = '{} -P {}'.format(scp_cmd, scp_port)

        scp_cmd = '{} "{}" {}:~/'.format(scp_cmd, out_path, scp_dst)

        print('\nrunning: {}\n'.format(scp_cmd))
        os.system(scp_cmd)
        rm_cmd = 'rm "{}"'.format(out_path)
        print('\nrunning: {}\n'.format(rm_cmd))
        os.system(rm_cmd)
    elif move_to_home:
        mv_cmd = 'mv "{:s}" ~'.format(out_path)
        print('\nrunning: {}\n'.format(mv_cmd))
        os.system(mv_cmd)

    if out_size > 1000:
        out_size /= 1000
        print('out_size:\n {} MB'.format(out_size))
    else:
        print('out_size:\n {} KB'.format(out_size))

    print('out_name:\n {}'.format(out_name))

    # import pyperclip
    # try:
    #     pyperclip.copy(out_name)
    #     spam = pyperclip.paste()
    # except pyperclip.PyperclipException as e:
    #     print('Copying to clipboard failed: {}'.format(e))
