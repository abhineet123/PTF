import csv
import os

root_dir = '../Datasets/DFT'
seq_name = 'atlas_moving_camera'
ext = 'jpg'

seq_dir = '{:s}/{:s}'.format(root_dir, seq_name)
csv_fname = '{:s}/{:s}.txt'.format(root_dir, seq_name)
if not os.path.isfile(csv_fname):
    print 'Ground truth CSV file {:s} does not exist'.format(csv_fname)
    exit()
with open(csv_fname, 'rb') as csvfile:
    gt_reader = csv.DictReader(csvfile, delimiter=',')
    frame_id = 1
    for row in gt_reader:
        old_fname = '{:s}/{:s}'.format(seq_dir, row['name'])
        if not os.path.isfile(old_fname):
            print 'File {:s} does not exist'.format(old_fname)
            exit()
        new_fname = '{:s}/frame{:05d}.{:s}'.format(seq_dir, frame_id, ext)
        frame_id += 1
        os.rename(old_fname, new_fname)
