from Misc import getParamDict
from Misc import readTrackingDataMOT
from Misc import writeCornersMOT
from Misc import getFileList
import os
import sys
import cv2
import shutil
import numpy as np

if __name__ == '__main__':

    params_dict = getParamDict()
    actors = params_dict['actors']
    sequences = params_dict['sequences']

    fix_frame_ids = 1
    swap_coords = 0
    swap_dimensions = 0
    invert_x = 0
    invert_y = 0
    # remove background detections
    remove_background = 0
    # minimum intensity difference between input and background images to be considered foreground
    bkg_thresh = 25
    # minimum ratio of foreground to background for detection to be retained
    frg_ratio_thresh = 0.50
    save_frg_mask = 1

    remove_oversized = 0
    height_thresh = 300
    width_thresh = 300
    area_thresh = 90000

    xmin_thresh = 0
    ymin_thresh = 0

    conf_thresh = -1

    replace_orig = 0

    actor_id = 2
    seq_id = 2
    # actor = None
    # seq_name = None
    actor = 'GRAM'
    # seq_name = 'M-30-HD'
    # seq_name = 'M-30'
    seq_name = 'Urban1'

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
        actor = actors[actor_id]
    if seq_name is None:
        sequences = sequences[actor]
        if seq_id >= len(sequences):
            print 'Invalid seq_id: ', seq_id
            sys.exit()
        seq_name = sequences[seq_id]

    in_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    detections = readTrackingDataMOT(in_fname)
    if detections is None:
        exit(0)
    print 'detections.shape: ', detections.shape

    if replace_orig:
        out_fname = in_fname
        print 'Replacing the original file with the corrected one'
    else:
        out_fname = db_root_dir + '/' + actor + '/' + seq_name + '_corr.txt'
        print 'Writing corrected detections to {:s}'.format(out_fname)

    out_file = open(out_fname, 'w')

    # sort by frame IDs
    print 'Sorting detections by frame IDs'
    detections = detections[detections[:,0].argsort()]
    print 'Done'


    n_detections = len(detections)
    n_frames = int(detections[-1, 0])
    if fix_frame_ids:
        n_frames += 1

    if invert_x or invert_y or remove_background:
        src_fname = db_root_dir + '/' + actor + '/Images/' + seq_name + '/' + img_name_fmt
        cap = cv2.VideoCapture()
        if not cap.open(src_fname):
            print 'The video file ', src_fname, ' could not be opened'
            sys.exit()
        ret, img = cap.read()
        img_height = img.shape[0]
        img_width = img.shape[1]
        print 'img_height: ', img_height
        print 'img_width: ', img_width

    frame_id = 1

    if remove_background:
        bkg_fname = db_root_dir + '/' + actor + '/Images/' + seq_name + '_bkg.jpg'
        bkg_img = cv2.imread(bkg_fname)
        bkg_img_gs = cv2.cvtColor(bkg_img, cv2.COLOR_BGR2GRAY)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_img = cv2.absdiff(img_gs, bkg_img_gs)
        frg_mask = sub_img > bkg_thresh
        cv2.imshow('Subtracted image', sub_img.astype(np.uint8))
        frg_mask_disp = frg_mask.astype(np.uint8)*255
        cv2.imshow('Foreground Mask', frg_mask_disp)
        if save_frg_mask:
            frg_mask_dir = db_root_dir + '/' + actor + '/Images/' + seq_name + '_frg_mask_{:d}'.format(bkg_thresh)
            if not os.path.exists(frg_mask_dir):
                os.makedirs(frg_mask_dir)
            frg_mask_fname = frg_mask_dir + '/' + 'image{:06d}.jpg'.format(frame_id)
            cv2.imwrite(frg_mask_fname, frg_mask_disp)

    print 'actor: ', actor
    print 'seq_name:', seq_name
    print 'n_detections: ', n_detections
    print 'n_frames: ', n_frames

    n_skipped_detections = 0

    for det_id in xrange(n_detections):
        curr_frame_id = int(detections[det_id][0])
        obj_id = int(detections[det_id][1])
        x = float(detections[det_id][2])
        y = float(detections[det_id][3])
        width = float(detections[det_id][4])
        height = float(detections[det_id][5])
        conf = float(detections[det_id][6])

        pt_x = float(detections[det_id][7])
        pt_y = float(detections[det_id][8])
        pt_z = float(detections[det_id][9])

        area = width * height

        if fix_frame_ids:
            curr_frame_id += 1

        if swap_coords:
            out_x = y
            out_y = x
        else:
            out_x = x
            out_y = y

        if invert_x:
            out_x = img_width - out_x

        if invert_y:
            out_y = img_height - out_y

        if swap_dimensions:
            out_width = height
            out_height = width
        else:
            out_width = width
            out_height = height

        if remove_oversized:
            if width_thresh > 0 and out_width > width_thresh:
                n_skipped_detections += 1
                print 'Skipping detection {:d} as its width {:f} is too large'.format(det_id, out_width)
                continue
            if height_thresh > 0 and out_height > height_thresh:
                n_skipped_detections += 1
                print 'Skipping detection {:d} as its height {:f} is too large'.format(det_id, out_height)
                continue
            if area_thresh > 0 and area > area_thresh:
                n_skipped_detections += 1
                print 'Skipping detection {:d} as its area {:f} is too large'.format(det_id, area)
                continue

        xmin_thresh = 0
        if ymin_thresh > 0:
            ymax = y + height
            if ymax < ymin_thresh:
                print 'Skipping detection {:d} as its ymax {:f} is too small'.format(det_id, ymax)
                continue

        if xmin_thresh > 0:
            xmax = x + width
            if xmax < xmin_thresh:
                print 'Skipping detection {:d} as its xmax {:f} is too small'.format(det_id, xmax)
                continue

        if conf_thresh > 0 and conf < conf_thresh:
            print 'Skipping detection {:d} as its confidence {:f} is too low'.format(det_id, conf)

        if remove_background:
            if curr_frame_id > frame_id:
                ret, img = cap.read()
                if not ret:
                    print 'Input image could not be read'
                    exit(0)
                img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sub_img = cv2.absdiff(img_gs, bkg_img_gs)
                frg_mask = sub_img > bkg_thresh
                frame_id = curr_frame_id
                frg_mask_disp = frg_mask.astype(np.uint8)*255
                if save_frg_mask:
                    frg_mask_fname = frg_mask_dir + '/' + 'image{:06d}.jpg'.format(frame_id)
                    cv2.imwrite(frg_mask_fname, frg_mask_disp)

            cv2.imshow('Subtracted image', sub_img.astype(np.uint8))
            cv2.imshow('Foreground Mask', frg_mask_disp)
            cv2.waitKey(1)

            min_x = int(max(out_x, 0))
            min_y = int(max(out_y, 0))
            max_x = int(min(out_x + out_width, img_width - 1))
            max_y = int(min(out_y + out_height, img_height - 1))
            frg_mask_patch = frg_mask[min_y:max_y + 1, min_x:max_x + 1]

            frg_ratio = float(np.count_nonzero(frg_mask_patch))/float(frg_mask_patch.size)
            if frg_ratio < frg_ratio_thresh:
                print 'removing detection {:d} in frame {:d} with foreground ratio {:f}'.format(
                    det_id, frame_id, frg_ratio)
                continue

        corr_detection = [curr_frame_id, obj_id, out_x, out_y, out_width, out_height, conf, pt_x, pt_y, pt_z]
        writeCornersMOT(out_file, corr_detection)
    out_file.close()
    if n_skipped_detections > 0:
        print '{:d} detections skipped'.format(n_skipped_detections)















