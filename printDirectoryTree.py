import os
import sys
import pyperclip
from datetime import datetime

from Misc import processArguments, write

params = {
    'start_path': '.',
    'markdown_mode': 0,
    'exts_to_include': [],
    'strings_to_exclude': [],
    'fix_weird_text': 1,
}
processArguments(sys.argv[1:], params)
start_path = params['start_path']
markdown_mode = params['markdown_mode']
exts_to_include = params['exts_to_include']
strings_to_exclude = params['strings_to_exclude']
fix_weird_text = params['fix_weird_text']

time_stamp = datetime.now().strftime("%y%m%d %H%M%S")

if exts_to_include:
    print('Excluding files with extension not in: {}'.format(exts_to_include))

if strings_to_exclude:
    print('Excluding paths containing: {}'.format(strings_to_exclude))

if markdown_mode:
    print('Using markdown mode')
    out_text = 'Updated on: {}\n'.format(time_stamp)
else:
    out_text = ''
for root, dirs, files in os.walk(start_path):
    if strings_to_exclude and any([k in root for k in strings_to_exclude]):
        continue
    level = root.replace(start_path, '').count(os.sep) - 1
    _root = os.path.relpath(root, start_path)
    if _root != '.':
        if markdown_mode:
            indent = '    ' * (level)
            _text = '{}- {}/\n'.format(indent, os.path.basename(root))
            out_text += _text
        else:
            _text = _root.replace(os.sep, '\t') + '\n'
            out_text += _text
    if markdown_mode:
        # write(_text)
        subindent = '    ' * (level + 1)
    else:
        subindent = _root.replace(os.sep, '\t')
    for f in files:
        f_ext = os.path.splitext(f)[1][1:]
        if exts_to_include and f_ext not in exts_to_include:
            continue
        if strings_to_exclude and any([k in f for k in strings_to_exclude]):
            continue
        if markdown_mode:
            _text = '{}- {}\n'.format(subindent, f)
        else:
            _text = '{}\t{}\n'.format(subindent, f)
        out_text += _text

if fix_weird_text:
    out_text = out_text.replace('?', 'fi')

try:
    pyperclip.copy(out_text)
    spam = pyperclip.paste()
except pyperclip.PyperclipException as e:
    print('Copying to clipboard failed: {}'.format(e))
else:
    print('Directory tree copied to clipboard')
    


