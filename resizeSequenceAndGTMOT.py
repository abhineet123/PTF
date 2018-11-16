from Misc import getParamDict
from Misc import readTrackingDataMOT
from Misc import writeCornersMOT
from Misc import getFileList
import os
import sys
import shutil
import cv2
import numpy as np

if __name__ == '__main__':

    params_dict = getParamDict()
    mot_actors = params_dict['mot_actors']
    mot_sequences = params_dict['mot_sequences']

    resize_images = 0
    resize_mult_factor = 1.0/1.5
    # n_split_seq = 15
    # n_split_seq = 30

    actor_id = 2
    seq_id = 1

    seq_type_id = 0

    actor = None
    seq_name = None
    # actor = 'GRAM'
    # seq_name = 'M-30'
    # seq_name = 'M-30-HD'
    # seq_name = 'Urban1'


    split_frames = None
    # split_frames = [1000, 1884, 2917, 3878, 4885, 5800]
    # split_frames = [805, 1568]
    # split_frames = [828, 1639,2487]

    # db_root_dir = '../Datasets'
    db_root_dir = 'C:/Datasets'

    # img_name_fmt = 'frame%05d.jpg'
    img_name_fmt = 'image%06d.jpg'
    img_ext = '.jpg'

    arg_id = 1
    if len(sys.argv) > arg_id:
        actor_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1

    if actor is None:
        actor = mot_actors[actor_id]
    if seq_name is None:
        sequences = mot_sequences[actor]
        if isinstance(sequences[0], list):
            sequences = sequences[seq_type_id]
        if seq_id >= len(sequences):
            print 'Invalid dataset_id: ', seq_id
            sys.exit()
        seq_name = sequences[seq_id]

    if resize_mult_factor < 1:
        out_prefix = '-Small'
    else:
        out_prefix = '-Large'

    gt_in_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    gt_in = readTrackingDataMOT(gt_in_fname)
    if gt_in is None:
        exit(0)
    n_gt_lines = len(gt_in)

    print 'actor: ', actor
    print 'seq_id:', seq_id, 'seq_name:', seq_name
    n_gt_frames = int(gt_in[-1][0])

    print 'Resizing sequence: {:s} by a factor of {:f}'.format(seq_name, resize_mult_factor)

    if resize_images:
        # cap = cv2.VideoCapture()
        # if not cap.open(src_fname):
        # print 'The image sequence ', src_fname, ' could not be opened'
        #     sys.exit()
        in_img_dir = db_root_dir + '/' + actor + '/' + seq_name
        img_list = getFileList(in_img_dir, img_ext)
        n_frames = len(img_list)
        print 'n_gt_frames: ', n_gt_frames

        if n_frames != n_gt_frames:
            msg = 'No. of frames in GT: {:d} does not match the number of images in the sequence: {:d}'.format(
                n_gt_frames, n_frames)
            raise SyntaxError(msg)
        in_img_template = in_img_dir + '/' + img_name_fmt
        out_img_dir = db_root_dir + '/' + actor + '/' + seq_name + out_prefix
        print 'Writing resized images to: {:s}'.format(out_img_dir)
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)
        out_img_template = out_img_dir + '/' + img_name_fmt
        for frame_id in xrange(n_gt_frames):
            in_fname = '{:s}/image{:06d}.jpg'.format(in_img_dir, frame_id + 1)
            # print in_fname
            in_img = cv2.imread(in_fname)
            # print in_img.shape
            out_img = cv2.resize(in_img, None, fx=resize_mult_factor,
                                 fy=resize_mult_factor,
                                 interpolation=cv2.INTER_LANCZOS4)
            # out_fname = out_img_template.format(frame_id + 1)
            out_fname = '{:s}/image{:06d}.jpg'.format(out_img_dir, frame_id + 1)
            cv2.imwrite(out_fname, out_img)
            if (frame_id + 1) % 100 == 0:
                print 'Done {:d} frames'.format(frame_id + 1)

    gt_out_fname = db_root_dir + '/' + actor + '/' + seq_name + out_prefix + '.txt'
    gt_out_fid = open(gt_out_fname, 'w')

    print 'Writing resized GT to: {:s}'.format(gt_out_fname)
    for gt_id in xrange(n_gt_lines):
        curr_frame_id = int(gt_in[gt_id][0])
        obj_id = int(gt_in[gt_id][1])
        x = float(gt_in[gt_id][2])
        y = float(gt_in[gt_id][3])
        width = float(gt_in[gt_id][4])
        height = float(gt_in[gt_id][5])
        conf = float(gt_in[gt_id][6])

        pt_x = float(gt_in[gt_id][7])
        pt_y = float(gt_in[gt_id][8])
        pt_z = float(gt_in[gt_id][9])

        out_x = x * resize_mult_factor
        out_y = y * resize_mult_factor
        out_width = width * resize_mult_factor
        out_height = height * resize_mult_factor

        curr_gt_out = [curr_frame_id, obj_id, out_x, out_y, out_width, out_height, conf, pt_x, pt_y, pt_z]
        writeCornersMOT(gt_out_fid, curr_gt_out)
        if (gt_id + 1) % 100 == 0:
            print 'Done {:d} lines'.format(gt_id + 1)
    gt_out_fid.close()













