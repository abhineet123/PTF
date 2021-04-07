import os
import shutil

import paramparse

# _params = {
#     'start_out_id': 0,
#     # 'start_out_id': 47,
#     # 'root_dir': 'G:/Datasets/MOT2015',
#     # 'root_dir': 'G:/Datasets/MOT2017',
#     'root_dir': '/data/CTMC',
#     # 'db_dir': '2DMOT2015',
#     'db_dir': '',
#     'db_type': 'train',
#     # 'db_type': 'test',
#     'det_type': '',
#     # 'det_type': 'FRCNN',
#     'ignore_img': 0,
#     'ignore_det': 1,
#     'no_move': 0,
#     'process_tra': 1,
# }


class Params:
    def __init__(self):
        self.cfg = ('',)
        self.db_dir = ''
        self.db_type = 'train'
        self.det_type = ''
        self.ignore_det = 1
        self.ignore_img = 0
        self.no_move = 0
        self.process_tra = 1
        self.root_dir = '/data/CTMC'
        self.start_out_id = 0

_params = Params()

paramparse.process(_params)

root_dir = _params['root_dir']
db_dir = _params['db_dir']
db_type = _params['db_type']
det_type = _params['det_type']
ignore_img = _params['ignore_img']
ignore_det = _params['ignore_det']
no_move = _params['no_move']
start_out_id = _params['start_out_id']
process_tra = _params['process_tra']

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

for seq_id, seq_name in enumerate(sequences):

    if process_tra:
        in_tra_path = os.path.join(in_img_root, seq_name, 'TRA', 'man_track.txt')
        assert os.path.exists(in_tra_path), "in_tra_path: {} does not exist".format(in_tra_path)
        out_tra_path = os.path.join(out_gt_root, seq_name + '.tra')

        print('{} --> {}'.format(in_tra_path, out_tra_path))
        if not no_move:
            shutil.move(in_tra_path, out_tra_path)

        continue

    if not ignore_det:
        if det_type:
            in_det_path = os.path.join(in_img_root, det_type, seq_name + '-' + det_type, 'det', 'det.txt')
        else:
            in_det_path = os.path.join(in_img_root, seq_name, 'det', 'det.txt')
        assert os.path.exists(in_det_path), "in_det_path: {} does not exist".format(in_det_path)
        out_det_path = os.path.join(out_det_root, seq_name + '.txt')
        if not no_move:
            shutil.move(in_det_path, out_det_path)
        print('{} --> {}'.format(in_det_path, out_det_path))

    if not ignore_img:
        in_img_path = os.path.join(in_img_root, seq_name, 'img1')

        _src_files = [os.path.join(in_img_path, k) for k in os.listdir(in_img_path) if
                      os.path.splitext(k.lower())[1] in img_exts]
        n_src_files = len(_src_files)
        n_frames_list.append(n_src_files)
        print('n_src_files: {}'.format(n_src_files))
        out_txt += "{}: ('{}', {}), \n".format(seq_id + start_out_id, seq_name, n_src_files)

        out_img_path = os.path.join(out_img_root, seq_name)
        if not no_move:
            assert os.path.exists(in_img_path), "in_img_path: {} does not exist".format(in_img_path)
            shutil.move(in_img_path, out_img_path)
        print('{} --> {}'.format(in_img_path, out_img_path))

    in_gt_path = os.path.join(in_img_root, seq_name, 'gt', 'gt.txt')
    if os.path.exists(in_gt_path):
        out_gt_path = os.path.join(out_gt_root, seq_name + '.txt')
        if not no_move:
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
