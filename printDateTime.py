from datetime import datetime
import sys, os
from Misc import processArguments
import pyperclip

if __name__ == '__main__':
    params = {
        'prefix': '',
        'file_path': '',
    }
    processArguments(sys.argv[1:], params)
    prefix = params['prefix']
    file_path = params['file_path']

    out_str = datetime.now().strftime("%y%m%d_%H%M%S")

    if prefix:
        out_str = '{}_{}'.format(prefix, out_str)

    sys.stdout.write('{:s}\n'.format(out_str))
    sys.stdout.flush()

    try:
        pyperclip.copy('_' + out_str)
        spam = pyperclip.paste()
    except pyperclip.PyperclipException as e:
        pass

    if file_path:
        filename_no_ext, _ext = os.path.splitext(os.path.basename(file_path))
        out_filename = '{}_{}{}'.format(filename_no_ext, out_str, _ext)
        out_file_path = os.path.join(os.path.basename(file_path), out_filename)
        os.rename(file_path, out_file_path)



