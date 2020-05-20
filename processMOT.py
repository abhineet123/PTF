import xml.etree.cElementTree as ET
import os
import shutil
import sys
import glob

from Misc import processArguments
from Misc import getParamDict

_params = {
    # 'root_dir': 'G:/Datasets/MOT2015',
    'root_dir': 'G:/Datasets/MOT2017',
    # 'db_dir': '2DMOT2015',
    'db_dir': '',
    'db_type': 'train',
    # 'db_type': 'test',
    'det_type': '',
    # 'det_type': 'FRCNN',
    'ignore_img': 0,
}

processArguments(sys.argv[1:], _params)

root_dir = _params['root_dir']
db_dir = _params['db_dir']
db_type = _params['db_type']
det_type = _params['det_type']
ignore_img = _params['ignore_img']
# actor_id = _params['actor_id']
# start_id = _params['start_id']
# ignored_region_only = _params['ignored_region_only']
# end_id = _params['end_id']

# params = getParamDict()
# actors = params['mot_actors']
# sequences = params['mot_sequences']
#
# actor = actors[actor_id]
# actor_sequences = sequences[actor]
#
# if end_id <= start_id:
#     end_id = len(actor_sequences) - 1

print('root_dir: {}'.format(root_dir))
print('db_type: {}'.format(db_type))
# print('actor_id: {}'.format(actor_id))
# print('start_id: {}'.format(start_id))
# print('end_id: {}'.format(end_id))

# print('actor: {}'.format(actor))
# print('actor_sequences: {}'.format(actor_sequences))

out_img_root = os.path.join(root_dir, 'Images')
out_gt_root = os.path.join(root_dir, 'Annotations')
out_det_root = os.path.join(root_dir, 'Detections')

print('out_img_root: {}'.format(out_img_root))
print('out_gt_root: {}'.format(out_gt_root))
print('out_det_root: {}'.format(out_det_root))

if not os.path.isdir(out_img_root):
    os.makedirs(out_img_root)
if not os.path.isdir(out_gt_root):
    os.makedirs(out_gt_root)
if not os.path.isdir(out_det_root):
    os.makedirs(out_det_root)

n_frames_list = []

if db_dir:
    in_root_dir = os.path.join(root_dir, db_dir)
else:
    in_root_dir = root_dir

in_img_root = os.path.join(in_root_dir, db_type)

sequences = [k for k in os.listdir(in_img_root)
             if os.path.isdir(os.path.join(in_img_root, k))]

print('sequences: {}'.format(sequences))
img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff', '.webp')

out_txt = ''

for seq_name in sequences:
    in_img_path = os.path.join(in_img_root, seq_name, 'img1')
    in_gt_path = os.path.join(in_img_root, seq_name, 'gt', 'gt.txt')

    if det_type:
        in_det_path = os.path.join(in_img_root, det_type, seq_name + '-' + det_type, 'det', 'det.txt')
    else:
        in_det_path = os.path.join(in_img_root, seq_name, 'det', 'det.txt')

    if not ignore_img:
        assert os.path.exists(in_img_path), "in_img_path: {} does not exist".format(in_img_path)

        _src_files = [os.path.join(in_img_path, k) for k in os.listdir(in_img_path) if
                      os.path.splitext(k.lower())[1] in img_exts]
        n_src_files = len(_src_files)
        n_frames_list.append(n_src_files)
        print('n_src_files: {}'.format(n_src_files))
        out_txt += '{}\n'.format(n_src_files)

        out_img_path = os.path.join(out_img_root, seq_name)
        shutil.move(in_img_path, out_img_path)
        print('{} --> {}'.format(in_img_path, out_img_path))

    assert os.path.exists(in_det_path), "in_det_path: {} does not exist".format(in_det_path)
    out_det_path = os.path.join(out_det_root, seq_name + '.txt')
    shutil.move(in_det_path, out_det_path)

    print('{} --> {}'.format(in_det_path, out_det_path))

    if os.path.exists(in_gt_path):
        out_gt_path = os.path.join(out_gt_root, seq_name + '.txt')
        shutil.move(in_gt_path, out_gt_path)
        print('{} --> {}'.format(in_gt_path, out_gt_path))

    print()
print(out_txt)

try:
    import pyperclip

    pyperclip.copy(out_txt)
    spam = pyperclip.paste()
except BaseException as e:
    print('Copying to clipboard failed: {}'.format(e))
