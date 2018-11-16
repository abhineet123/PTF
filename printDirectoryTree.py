import os
import sys
import pyperclip

from Misc import processArguments, write

params = {
    'start_path': '.',
}
processArguments(sys.argv[1:], params)
start_path = params['start_path']
out_text = ''
for root, dirs, files in os.walk(start_path):
    level = root.replace(start_path, '').count(os.sep)
    # indent = '\t' * (level)
    # _text = '{}{}/\n'.format(indent, os.path.basename(root))
    _root = os.path.relpath(root, start_path)
    if _root != '.':
        _text = _root.replace(os.sep, '\t') + '\n'
        out_text += _text
    # write(_text)
    # subindent = '\t' * (level + 1)
    subindent = _root.replace(os.sep, '\t')
    for f in files:
        _text = '{}\t{}\n'.format(subindent, f)
        out_text += _text
        # write(_text)

try:
    pyperclip.copy(out_text)
    spam = pyperclip.paste()
except pyperclip.PyperclipException as e:
    print('Copying to clipboard failed: {}'.format(e))

