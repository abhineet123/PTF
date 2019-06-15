import os
import sys, shutil
from random import shuffle

from Misc import processArguments, sortKey

params = {
    'file_ext': 'eps',
    'folder_name': '.',
}
processArguments(sys.argv[1:], params)
file_ext = params['file_ext']
folder_name = params['folder_name']

src_files = [f for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f))]
if file_ext:
    src_files = [f for f in src_files if f.endswith(file_ext)]
src_files.sort(key=sortKey)

n_files = len(src_files)


for _f in src_files:
    cmd = 'epstopdf {}'.format(_f)
    print(cmd)
    os.system(cmd)


