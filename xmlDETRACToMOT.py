import xml.etree.cElementTree as ET
from Misc import getParamDict
import glob

root_dir = 'E:/Datasets'
actor_id = 4
start_id = 0
end_id = -1
ignored_region_only = 0

params = getParamDict()
actors = params['mot_actors']
sequences = params['mot_sequences']
actor = actors[actor_id]
n_frames_list = []

if end_id <= start_id:
    end_id = len(sequences[actor]) - 1

for seq_id in range(start_id, end_id + 1):
    seq_name = sequences[actor][seq_id]
    fname = '{:s}/{:s}/Annotations/{:s}.xml'.format(root_dir, actor, seq_name)
    tree = ET.parse(fname)
    root = tree.getroot()

    out_fname = '{:s}/{:s}/Annotations/{:s}.txt'.format(root_dir, actor, seq_name)
    out_fid = open(out_fname, 'w')

    for obj in tree.iter('ignored_region'):
        bndbox = obj.find('box')
        if bndbox is None:
            continue
        xmin = float(bndbox.attrib['left'])
        ymin = float(bndbox.attrib['top'])
        width = float(bndbox.attrib['width'])
        height = float(bndbox.attrib['height'])
        out_fid.write('-1,-1,{:f},{:f},{:f},{:f},1,-1,-1,-1\n'.format(
            xmin, ymin, width, height))

    if ignored_region_only:
        out_fid.close()
        continue

    img_dir = '{:s}/{:s}/Images/{:s}'.format(root_dir, actor, seq_name)
    n_frames = len(glob.glob('{:s}/*.jpg'.format(img_dir)))
    n_frames_list.append(n_frames)

    print('Processing sequence {:d} :: {:s}'.format(seq_id, seq_name))
    for frame_obj in tree.iter('frame'):
        target_list = frame_obj.find('target_list')
        frame_id = int(frame_obj.attrib['num'])
        for obj in target_list.iter('target'):
            bndbox = obj.find('box')
            obj_id = int(obj.attrib['id'])
            xmin = float(bndbox.attrib['left'])
            ymin = float(bndbox.attrib['top'])
            width = float(bndbox.attrib['width'])
            height = float(bndbox.attrib['height'])
            out_fid.write('{:d},{:d},{:f},{:f},{:f},{:f},1,-1,-1,-1\n'.format(
                frame_id, obj_id, xmin, ymin, width, height))
        if frame_id % 100 == 0:
            print('\t Done {:d}/{:d} frames'.format(frame_id, n_frames))
    out_fid.close()

print(n_frames_list)
