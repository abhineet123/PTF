import os
import sys
from Misc import processArguments, sortKey

params = {
    'file_ext': 'eps',
    'folder_name': '.',
    'recursive': 0,
}
processArguments(sys.argv[1:], params)
file_ext = params['file_ext']
folder_name = params['folder_name']
recursive = params['recursive']

if recursive:
    src_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                     f.endswith(file_ext)]
                    for (dirpath, dirnames, filenames) in os.walk(folder_name, followlinks=True)]
    src_files = [item for sublist in src_file_gen for item in sublist]
else:
    src_files = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
                 if os.path.isfile(os.path.join(folder_name, f)) and f.endswith(file_ext)]

src_files.sort(key=sortKey)

n_files = len(src_files)


for _f in src_files:
    cmd = 'epstopdf {}'.format(_f)
    print(cmd)
    os.system(cmd)


