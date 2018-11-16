import xml.etree.cElementTree as ET

root_dir = 'C:/Datasets/GRAM/Annotations'
seq_names = ['M-30', 'M-30-HD', 'Urban1']
seq_n_frames = [7520, 9390, 23435]
seq_id = 1
seq_name = seq_names[seq_id]
n_frames = seq_n_frames[seq_id]
out_fname= '{:s}/{:s}.txt'.format(root_dir, seq_name)
out_fid = open(out_fname, 'w')
for frame_id in xrange(n_frames):
    fname = '{:s}/{:s}/xml/{:d}.xml'.format(root_dir, seq_name, frame_id)
    tree = ET.parse(fname)
    root = tree.getroot()
    if (frame_id + 1) % 100 == 0:
        print 'frame {:d}/{:d}'.format(frame_id + 1, n_frames)
    for obj in tree.iter('object'):
        obj_id = int(obj.find('ID').text)
        occluded = int(obj.find('occluded').text)
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        width = xmax - xmin
        height = ymax - ymin
        # print '{:d}: {:d}, {:d}, {:d}, {:d}'.format(obj_id, xmin, ymin, xmax, ymax)
        out_fid.write('{:d},{:d},{:d},{:d},{:d},{:d},1,-1,-1,-1\n'.format(
           frame_id + 1, obj_id, xmin, ymin, width, height))
out_fid.close()
