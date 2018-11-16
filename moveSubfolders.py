import glob
from Misc import processArguments
import sys, os, shutil

params = {
    'src_dir': '.',
    'dst_dir': '',
}

processArguments(sys.argv[1:], params)
src_dir = params['src_dir']
dst_dir = params['dst_dir']

if not dst_dir:
    dst_dir = src_dir

src_dir = os.path.abspath(src_dir)
dst_dir = os.path.abspath(dst_dir)

subfolders = [os.path.abspath(f) for f in glob.iglob(src_dir + '/**/', recursive=True) if os.path.isdir(f) and
              'annotations' not in f and os.path.basename(f) != 'bin']

for src in subfolders:
    if src == src_dir or src == dst_dir:
        continue

    if os.path.dirname(src) == dst_dir:
        continue

    curr_subfolders = [f for f in os.listdir(src) if os.path.isdir(os.path.join(src, f)) and 'annotations' not in f]
    if curr_subfolders:
        continue

    print('moving {}'.format(src))
    try:
        shutil.move(src, dst_dir)
    except shutil.Error as e:
        print('Failure: {}'.format(e))
        continue
    except FileNotFoundError as e:
        print('Failure: {}'.format(e))
        continue
    except OSError as e:
        print('Failure: {}'.format(e))
        continue
