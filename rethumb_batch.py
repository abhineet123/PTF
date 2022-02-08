import sys
import os

from Misc import processArguments

params = {
    'thumbs_dir': 'C:\\Users\\Tommy\\AppData\\Roaming\\XYplorer\\Thumbnails',
    'out_dir': 'Z:\\rethumb',

}
processArguments(sys.argv[1:], params)
thumbs_dir = params['thumbs_dir']
out_dir = params['out_dir']


src_file_names = [f for f in os.listdir(thumbs_dir)
                  if os.path.isfile(os.path.join(thumbs_dir, f))
                  and f.endswith('.dat2')]

out_txt = ''
for src_fname in src_file_names:    
    filename, file_extension = os.path.splitext(src_fname)
    
    cmd = 'rethumb "{}", "{}", "{}";'.format(thumbs_dir, filename, out_dir)
    out_txt += cmd

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
try:
    import pyperclip
    pyperclip.copy(out_txt)
    spam = pyperclip.paste()
except BaseException as e:
    print('Copying to clipboard failed: {}'.format(e))