import sys
import os
import argparse

def parserToParams(parser: argparse.ArgumentParser,
                   class_name='Params'):
    """

    :param argparse.ArgumentParser parser:
    :return:
    """
    optionals = parser._optionals._option_string_actions
    positionals = parser._positionals._option_string_actions
    try:
        all_params = {**optionals, **positionals}
    except:
        all_params = optionals.copy()
        all_params.update(positionals)

    all_params_names = sorted(list(all_params.keys()))

    header_text = 'class {}:\n'.format(class_name)
    out_text = '\tdef __init__(self):\n'
    help_text = '\t\tself.help = {\n'
    doc_text = '\t"""\n'

    for _name in all_params_names:
        __name = _name[2:]
        if not __name or _name == '--h' or _name == '--help':
            continue
        _param = all_params[_name]
        _help = _param.help

        if isinstance(_param.default, str):
            default_str = "'{}'".format(_param.default)
        else:
            default_str = '{}'.format(_param.default)

        out_text += '\t\tself.{}={}\n'.format(__name, default_str)
        help_text += "\t\t\t'{}': '{}',\n".format(__name, _help)

        doc_text += '\t:param {} {}: {}\n'.format(type(_param.default).__name__, __name, _help)

    help_text += "\t\t}"
    doc_text += '\t"""\n'

    out_text += help_text

    out_text = header_text + doc_text + out_text
    out_fname = '{}.py'.format(class_name)

    out_path = os.path.abspath(out_fname)

    print('Writing output to {}'.format(out_path))
    with open(out_path, 'w') as fid:
        fid.write(out_text)


def dictToParams(param_dict: dict, class_name='Params'):
    """

    :param dict parser:
    :return:
    """

    all_params_names = sorted(list(param_dict.keys()))

    header_text = 'class {}:\n'.format(class_name)
    out_text = '\tdef __init__(self):\n'
    help_text = '\t\tself.help = {\n'
    doc_text = '\t"""\n'

    for _name in all_params_names:
        default = param_dict[_name]
        _help = ''

        if isinstance(default, str):
            default_str = "'{}'".format(default)
        else:
            default_str = '{}'.format(default)

        out_text += '\t\tself.{}={}\n'.format(_name, default_str)
        help_text += "\t\t\t'{}': '{}',\n".format(_name, _help)

        doc_text += '\t:param {} {}: {}\n'.format(type(default).__name__, _name, _help)

    help_text += "\t\t}"
    doc_text += '\t"""\n'

    out_text += help_text

    out_text = header_text + doc_text + out_text
    out_fname = '{}.py'.format(class_name)

    out_path = os.path.abspath(out_fname)

    print('Writing output to {}'.format(out_path))
    with open(out_path, 'w') as fid:
        fid.write(out_text)