# Mouse interface:

# clicking on the patch can select either the entire patch or a specific corner:
# entire patch is selected if the clicked point is nearest to its centroid
# otherwise the nearest corner is selected
# hold left button and drag to move the patch/corner around
# release left button to accept changes
# right click to go to next frame

# the corner selected in the current frame remains selected in the next frame too

# Keyboard interface:

# p/space : toggle pause
# 0: select entire patch
# 1/2/3/4: select ul/ur/ll/lr corner respectively
# +/- : increase decrease unit of translation
# s / down arrow : move patch or corner down
# w / up arrow :  move patch or corner up
# d / right arrow :  move patch or corner right
# a / left arrow :  move patch or corner left
# n: accept changes (if any) and move to next frame
# r: save changes and go to previous frame
# c : load corners from corrected ground truth
# g: load corners from original ground truth
# G : read all corners from the original ground truth

from Misc import readTrackingData
from Misc import getParamDict
from Misc import readDistGridParams
from Misc import drawRegion
from Misc import writeCorners
from Misc import getTrackingObject2
from Misc import getNormalizedUnitSquarePts
from Misc import getPixVals
from Misc import stackImages

import sys
import cv2
import numpy as np
import os
import glob
import math
import shutil

import utility as util

curr_img = None
drawn_img = None

curr_corners = None
curr_pts = None
curr_patch = None
curr_patch_resized = None

curr_pt_id = 0

left_click_location = None
obj_selected_mouse = False
obj_selected_kb = False

final_location = None
final_location_selected_mouse = True
final_location_selected_kb = True
title = None


def translateCorners(corners, trans_x, trans_y):
    corners[0, :] += trans_x
    corners[1, :] += trans_y
    return corners


def drawLines(img, col=(0, 0, 255), hover_pt=None):
    if len(curr_corners) == 0:
        cv2.imshow(title, img)
        return
    for i in xrange(len(curr_corners) - 1):
        cv2.line(img, curr_corners[i], curr_corners[i + 1], col, 1)


def getNearestCorner(pt):
    centroid = np.mean(curr_corners, axis=1)
    print 'centroid: ', centroid

    diff_x = centroid[0] - pt[0]
    diff_y = centroid[1] - pt[1]

    min_dist = diff_x * diff_x + diff_y * diff_y
    min_dist_id = -1
    for i in xrange(0, 4):
        diff_x = curr_corners[0, i] - pt[0]
        diff_y = curr_corners[1, i] - pt[1]

        curr_dist = diff_x * diff_x + diff_y * diff_y
        if curr_dist < min_dist:
            min_dist = curr_dist
            min_dist_id = i
    return min_dist_id


def mouseHandler(event, x, y, flags=None, param=None):
    global left_click_location, obj_selected_mouse
    global final_location, final_location_selected_mouse
    global curr_corners, curr_pt_id
    global curr_img, drawn_img, curr_img_gs
    global curr_pts, curr_patch, curr_patch_resized
    global std_corners, std_pts_hm, corners_changed
    global pause_seq, from_last_frame, to_next_frame
    global border_size

    if border_size > 0:
        x = (x - border_size)
        y = (y - border_size)

    if event == cv2.EVENT_LBUTTONDOWN:
        left_click_location = [x, y]
        final_location_selected_mouse = not final_location_selected_mouse
        if not final_location_selected_mouse:
            print 'Patch selected for mouse modification'
            obj_selected_mouse = True
            pause_seq = True
            curr_pt_id = getNearestCorner(left_click_location)
        else:
            print 'Patch position confirmed'
        return
    elif event == cv2.EVENT_LBUTTONUP:
        final_location = [x, y]
        final_location_selected_mouse = not final_location_selected_mouse
    elif event == cv2.EVENT_RBUTTONDOWN:
        to_next_frame = True
        pause_seq = False
    elif event == cv2.EVENT_RBUTTONUP:
        pass
    elif event == cv2.EVENT_MBUTTONDOWN:
        pass
    elif event == cv2.EVENT_MOUSEMOVE:
        if final_location_selected_mouse:
            return
        # print 'mouse hover detected'
        # if len(pts) == 0:
        # return
        if curr_pt_id == -1:
            centroid = np.mean(curr_corners, axis=1)
            centroid_disp_x = x - centroid[0]
            centroid_disp_y = y - centroid[1]
            curr_corners[0, :] += centroid_disp_x
            curr_corners[1, :] += centroid_disp_y
        else:
            curr_corners[0, curr_pt_id] = x
            curr_corners[1, curr_pt_id] = y
        corners_changed = True
        from_last_frame = True


def mouseHandlerResized(event, x_resized, y_resized, flags=None, param=None):
    global left_click_location, obj_selected_mouse
    global final_location, final_location_selected_mouse
    global curr_corners, curr_pt_id
    global curr_img, drawn_img, curr_img_gs
    global curr_pts, curr_patch, curr_patch_resized
    global std_corners, std_pts_hm, corners_changed
    global pause_seq, from_last_frame, to_next_frame
    global img_resize_factor, border_size


    x = x_resized / img_resize_factor
    y = y_resized / img_resize_factor

    if border_size > 0:
        x = (x - border_size)
        y = (y - border_size)


    if event == cv2.EVENT_LBUTTONDOWN:
        left_click_location = [x, y]
        final_location_selected_mouse = not final_location_selected_mouse
        if not final_location_selected_mouse:
            print 'Patch selected for mouse modification'
            obj_selected_mouse = True
            pause_seq = True
            curr_pt_id = getNearestCorner(left_click_location)
        else:
            print 'Patch position confirmed'
        return
    elif event == cv2.EVENT_LBUTTONUP:
        final_location = [x, y]
        final_location_selected_mouse = not final_location_selected_mouse
    elif event == cv2.EVENT_RBUTTONDOWN:
        to_next_frame = True
        pause_seq = False
    elif event == cv2.EVENT_RBUTTONUP:
        pass
    elif event == cv2.EVENT_MBUTTONDOWN:
        pass
    elif event == cv2.EVENT_MOUSEMOVE:
        if final_location_selected_mouse:
            return
        # print 'mouse hover detected'
        # if len(pts) == 0:
        # return
        if curr_pt_id == -1:
            centroid = np.mean(curr_corners, axis=1)
            centroid_disp_x = x - centroid[0]
            centroid_disp_y = y - centroid[1]
            curr_corners[0, :] += centroid_disp_x
            curr_corners[1, :] += centroid_disp_y
        else:
            curr_corners[0, curr_pt_id] = x
            curr_corners[1, curr_pt_id] = y
        corners_changed = True
        from_last_frame = True


if __name__ == '__main__':

    params_dict = getParamDict()
    # param_ids = readDistGridParams()
    pause_seq = 1
    gt_col = (0, 255, 0)
    text_col = (0, 255, 0)
    actors = params_dict['actors']
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d'
    img_name_ext = 'jpg'
    res_x = 100
    res_y = 100
    res_from_size = 0
    resize_factor = 4
    trans_unit = 0.1
    show_error = 0
    show_patches = 1
    overwrite_gt = 0
    show_resized_img = 1
    img_resize_factor = 3
    stack_order = 0  # 0: row major 1: column major
    show_init = 0
    border_size = 0
    border_col = (255, 255, 255)

    show_error = show_error and show_patches

    actor_id = 3
    seq_id = 25
    # should be the id of the last frame for which ground truth does not need adjustment;
    # note that frame id is zero base so should be one less than the frame number in the filename
    init_frame_id = 0

    actor = actors[actor_id]
    sequences = sequences[actor]
    seq_name = sequences[seq_id]

    from_last_frame = False
    manual_init = False

    print 'actor: ', actor
    print 'seq_name: ', seq_name

    img_name_fmt = '{:s}.{:s}'.format(img_name_fmt, img_name_ext)
    src_dir = db_root_dir + '/' + actor + '/' + seq_name
    src_fname = src_dir + '/' + img_name_fmt

    no_of_frames = len(glob.glob1(src_dir, '*.{:s}'.format(img_name_ext)))
    # no_of_frames = len([name for name in os.listdir(src_dir) if os.path.isfile(name)])
    print 'no_of_frames in source directory: ', no_of_frames

    # cap = cv2.VideoCapture()
    # if not cap.open(src_fname):
    # print 'The video file ', src_fname, ' could not be opened'
    # sys.exit()
    # ret, init_img = cap.read()

    fname = '{:s}/frame{:05d}.jpg'.format(src_dir, init_frame_id + 1)
    print 'fname: ', fname

    init_img = cv2.imread(fname)
    if len(init_img.shape) == 2:
        init_img = cv2.cvtColor(init_img, cv2.COLOR_GRAY2RGB)
    gt_corners_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    if overwrite_gt:
        corrected_corners_fname = gt_corners_fname
    else:
        corrected_corners_fname = db_root_dir + '/' + actor + '/' + seq_name + '_corr.txt'
    if not os.path.isfile(gt_corners_fname) or manual_init:
        print 'ground truth file not found. Using manual object initialization instead'.format(gt_corners_fname)
        sel_pts = getTrackingObject2(init_img)
        init_corners = np.asarray(sel_pts).astype(np.float64).T
        from_last_frame = True
        print 'init_corners:', init_corners
        pause_seq = True
        gt_frames = no_of_frames
        ground_truth = np.empty([1, 8])
        ground_truth[0] = init_corners.flatten(order='F')
        manual_init = True
    else:
        ground_truth = readTrackingData(gt_corners_fname)
        gt_frames = ground_truth.shape[0]
        print 'no_of_frames in ground truth: ', gt_frames
        if gt_frames != no_of_frames:
            print 'Mismatch between the no. of frames in the source directory and the ground truth'
        init_corners = np.asarray([ground_truth[init_frame_id, 0:2].tolist(),
                                   ground_truth[init_frame_id, 2:4].tolist(),
                                   ground_truth[init_frame_id, 4:6].tolist(),
                                   ground_truth[init_frame_id, 6:8].tolist()]).T
    if overwrite_gt and os.path.isfile(gt_corners_fname):
        backup_gt_fname = gt_corners_fname.replace('.txt', '.back_ptf')
        print 'Backing up existing GT to {:s}'.format(backup_gt_fname)
        shutil.move(gt_corners_fname, backup_gt_fname)
    templ_corners = np.asarray([ground_truth[0, 0:2].tolist(),
                                ground_truth[0, 2:4].tolist(),
                                ground_truth[0, 4:6].tolist(),
                                ground_truth[0, 6:8].tolist()]).T
    center_x = (templ_corners[0, 0] + templ_corners[0, 1] + templ_corners[0, 2] + templ_corners[0, 3]) / 4
    center_y = (templ_corners[1, 0] + templ_corners[1, 1] + templ_corners[1, 2] + templ_corners[1, 3]) / 4
    size_x = int((abs(templ_corners[0, 0] - center_x) + abs(templ_corners[0, 1] - center_x)
                  + abs(templ_corners[0, 2] - center_x) + abs(templ_corners[0, 3] - center_x)) / 2)
    size_y = int((abs(templ_corners[1, 0] - center_y) + abs(templ_corners[1, 1] - center_y)
                  + abs(templ_corners[1, 2] - center_y) + abs(templ_corners[1, 3] - center_y)) / 2)

    # size_x = int((abs(templ_corners[0, 1] - templ_corners[0, 0]) +
    # abs(templ_corners[0, 2] - templ_corners[0, 3])) / 2)
    # size_y = int((abs(templ_corners[1, 3] - templ_corners[1, 0]) +
    # abs(templ_corners[1, 2] - templ_corners[1, 1])) / 2)

    print 'Object size: {:d}x{:d}'.format(size_x, size_y)
    if res_from_size:
        res_x = int(size_x / resize_factor)
        res_y = int(size_y / resize_factor)

    n_pts = res_x * res_y
    patch_size_x = res_x * resize_factor
    patch_size_y = res_y * resize_factor

    std_pts, std_corners = getNormalizedUnitSquarePts(res_x, res_y, 0.5)
    std_pts_hm = util.homogenize(std_pts)

    init_hom_mat = np.mat(util.compute_homography(std_corners, init_corners))
    init_pts = util.dehomogenize(init_hom_mat * std_pts_hm)

    init_img_gs = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    img_height, img_width = init_img_gs.shape
    init_pixel_vals = np.mat([util.bilin_interp(init_img_gs, init_pts[0, pt_id], init_pts[1, pt_id]) for pt_id in
                              xrange(n_pts)])

    if init_frame_id == 0:
        show_init = 0

    if show_patches:
        init_patch = np.reshape(init_pixel_vals, (res_y, res_x)).astype(np.uint8)
        init_patch_resized = cv2.resize(init_patch, (patch_size_x, patch_size_y))
        drawRegion(init_img, init_corners, gt_col, 1, annotate_corners=False)
        if show_init:
            init_patch_win_name = 'Initial Patch'
            cv2.namedWindow(init_patch_win_name)
            cv2.imshow(init_patch_win_name, init_patch_resized)
            init_img_win_name = 'Initial Image'
            cv2.namedWindow(init_img_win_name)
            cv2.imshow(init_img_win_name, init_img)

        templ_fname = '{:s}/frame{:05d}.jpg'.format(src_dir, 1)
        templ_img = cv2.imread(templ_fname)
        if len(templ_img.shape) == 2:
            templ_img = cv2.cvtColor(templ_img, cv2.COLOR_GRAY2RGB)
        templ_img_gs = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        templ_hom_mat = np.mat(util.compute_homography(std_corners, templ_corners))
        templ_pts = util.dehomogenize(templ_hom_mat * std_pts_hm)
        templ_pixel_vals = np.mat(
            [util.bilin_interp(templ_img_gs, templ_pts[0, pt_id],
                               templ_pts[1, pt_id])
             for pt_id in xrange(n_pts)])
        templ_patch = np.reshape(templ_pixel_vals, (res_y, res_x)).astype(np.uint8)
        templ_patch_resized = cv2.resize(templ_patch, (patch_size_x, patch_size_y))
        # templ_patch_win_name = 'Template Patch'
        # cv2.namedWindow(templ_patch_win_name)
        # cv2.imshow(templ_patch_win_name, templ_patch_resized)

        drawRegion(templ_img, templ_corners, gt_col, 1)
        templ_img_win_name = 'Template'
        cv2.namedWindow(templ_img_win_name)
        cv2.imshow(templ_img_win_name, templ_img)

        curr_patch_win_name = 'Template & Current Patches'
        cv2.namedWindow(curr_patch_win_name)
        curr_pixel_vals = init_pixel_vals.copy()
        curr_patch_resized = init_patch_resized

    window_name = seq_name
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouseHandler)
    # if border_size > 0:
    #     bordered_window_name = window_name + ' Bordered'
    #     cv2.namedWindow(bordered_window_name)

    to_next_frame = False
    corners_changed = False

    curr_corners = init_corners
    curr_pts = init_pts
    curr_img = init_img.copy()
    drawn_img = curr_img.copy()
    curr_img_gs = init_img_gs

    if show_resized_img:
        resized_window_name = seq_name + ' Resized'
        cv2.namedWindow(resized_window_name)
        cv2.setMouseCallback(resized_window_name, mouseHandlerResized)
        # if border_size > 0:
        #     bordered_resized_window_name = resized_window_name + ' Bordered'
        #     cv2.namedWindow(bordered_resized_window_name)

    resized_img_shape = (curr_img.shape[1] * img_resize_factor, curr_img.shape[0] * img_resize_factor)

    frame_id = 0
    corrected_gt = []
    while frame_id <= init_frame_id:
        curr_corners = np.asarray([ground_truth[frame_id, 0:2].tolist(),
                                   ground_truth[frame_id, 2:4].tolist(),
                                   ground_truth[frame_id, 4:6].tolist(),
                                   ground_truth[frame_id, 6:8].tolist()]).T
        corrected_gt.append(curr_corners.copy())
        frame_id += 1

    frame_id -= 1
    while frame_id < no_of_frames:
        if frame_id == gt_frames:
            print 'End of ground truth reached'
            pause_seq = True
            from_last_frame = True

        if not pause_seq or frame_id == init_frame_id:
            if len(corrected_gt) >= frame_id:
                corrected_gt[frame_id - 1] = curr_corners.copy()
            else:
                corrected_gt.append(curr_corners.copy())

            frame_id += 1
            if not from_last_frame:
                curr_corners = np.asarray([ground_truth[frame_id - 1, 0:2].tolist(),
                                           ground_truth[frame_id - 1, 2:4].tolist(),
                                           ground_truth[frame_id - 1, 4:6].tolist(),
                                           ground_truth[frame_id - 1, 6:8].tolist()]).T
                curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners))
                curr_pts = util.dehomogenize(curr_hom_mat * std_pts_hm)
            # ret, curr_img = cap.read()
            # if not ret:
            # print 'End of sequence reached unexpectedly'
            # break
            curr_img = cv2.imread('{:s}/frame{:05d}.jpg'.format(src_dir, frame_id))
            if len(curr_img.shape) == 2:
                curr_img = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)
            curr_img_gs = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            # try:
            # if show_patches:
            # curr_pixel_vals = getPixVals(curr_pts, curr_img_gs)
            # curr_patch = np.reshape(curr_pixel_vals, (res_y, res_x)).astype(np.uint8)
            # curr_patch_resized = cv2.resize(curr_patch, (patch_size_x, patch_size_y))
            # except IndexError:
            # print 'curr_corners: \n', curr_corners
            # print 'curr_hom_mat: \n', curr_hom_mat
            #     print 'Invalid homography or out of range corners'
            if show_patches:
                curr_pixel_vals = getPixVals(curr_pts, curr_img_gs)
                curr_patch = np.reshape(curr_pixel_vals, (res_y, res_x)).astype(np.uint8)
                curr_patch_resized = cv2.resize(curr_patch, (patch_size_x, patch_size_y))
            drawn_img = curr_img.copy()
            fps_text = 'frame: {:4d}/{:4d}'.format(frame_id, no_of_frames)
            if show_error:
                error = math.sqrt(np.sum(np.square(curr_pixel_vals - init_pixel_vals)))
                fps_text = '{:s} error: {:f}'.format(fps_text, error)
            cv2.putText(drawn_img, fps_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, text_col)
            drawRegion(drawn_img, curr_corners, gt_col, 1)

        if to_next_frame:
            to_next_frame = False
            pause_seq = True

        key = cv2.waitKey(1)
        if key == 32 or key == ord('p'):
            pause_seq = not pause_seq
        elif key == ord('+'):
            trans_unit *= 2
            print 'Increased translation unit to: {:f}'.format(trans_unit)
        elif key == ord('-'):
            trans_unit /= 2
            print 'Decreased translation unit to: {:f}'.format(trans_unit)
        elif key == ord('1'):
            print 'Top left corner selected for KB modification'
            curr_pt_id = 0
            obj_selected_kb = True
            final_location_selected_kb = False
        elif key == ord('2'):
            print 'Top right corner selected for KB modification'
            curr_pt_id = 1
            obj_selected_kb = True
            final_location_selected_kb = False
        elif key == ord('3'):
            print 'Bottom left corner selected for KB modification'
            curr_pt_id = 2
            obj_selected_kb = True
            final_location_selected_kb = False
        elif key == ord('4'):
            print 'Bottom right corner selected for KB modification'
            curr_pt_id = 3
            obj_selected_kb = True
            final_location_selected_kb = False
        elif key == ord('0'):
            print 'Patch selected for KB modification'
            curr_pt_id = -1
            obj_selected_kb = True
            final_location_selected_kb = False
        elif key == ord('s') or key == 2621440:
            if curr_pt_id < 0:
                print 'Moving patch down'
                curr_corners[1, :] += trans_unit
            else:
                print 'Moving corner down'
                curr_corners[1, curr_pt_id] += trans_unit
            corners_changed = True
            from_last_frame = True
        elif key == ord('w') or key == 2490368:
            if curr_pt_id < 0:
                print 'Moving patch up'
                curr_corners[1, :] -= trans_unit
            else:
                print 'Moving corner up'
                curr_corners[1, curr_pt_id] -= trans_unit
            corners_changed = True
            from_last_frame = True
        elif key == ord('d') or key == 2555904:
            if curr_pt_id < 0:
                print 'Moving patch right'
                curr_corners[0, :] += trans_unit
            else:
                print 'Moving corner right'
                curr_corners[0, curr_pt_id] += trans_unit
            corners_changed = True
            from_last_frame = True
        elif key == ord('a') or key == 2424832:
            if curr_pt_id < 0:
                print 'Moving patch left'
                curr_corners[0, :] -= trans_unit
            else:
                print 'Moving corner left'
                curr_corners[0, curr_pt_id] -= trans_unit
            corners_changed = True
            from_last_frame = True
        elif key == ord('n') or key == ord('N'):
            to_next_frame = True
            pause_seq = False
        elif key == ord('r') or key == ord('R'):
            # go to previous frame
            if frame_id > 1:
                pause_seq = True
                frame_id -= 1
                curr_corners = corrected_gt[frame_id - 1]
                curr_img = cv2.imread('{:s}/frame{:05d}.jpg'.format(src_dir, frame_id))
                if len(curr_img.shape) == 2:
                    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)
                curr_img_gs = cv2.cvtColor(curr_img, cv2.cv.CV_BGR2GRAY).astype(np.float64)
                corners_changed = True
        elif key == ord('c'):
            if len(corrected_gt) >= frame_id:
                print 'Reading current corners from corrected ground truth'
                curr_corners = corrected_gt[frame_id - 1]
                corners_changed = True
        elif key == ord('g'):
            print 'Reading current corners from ground truth'
            curr_corners = np.asarray([ground_truth[frame_id - 1, 0:2].tolist(),
                                       ground_truth[frame_id - 1, 2:4].tolist(),
                                       ground_truth[frame_id - 1, 4:6].tolist(),
                                       ground_truth[frame_id - 1, 6:8].tolist()]).T
            corners_changed = True
        elif key == ord('G'):
            print 'Reading all corners from ground truth'
            curr_corners = np.asarray([ground_truth[frame_id - 1, 0:2].tolist(),
                                       ground_truth[frame_id - 1, 2:4].tolist(),
                                       ground_truth[frame_id - 1, 4:6].tolist(),
                                       ground_truth[frame_id - 1, 6:8].tolist()]).T
            from_last_frame = False
            corners_changed = True
        elif key == 27:
            break

        if corners_changed:
            corners_changed = False
            try:
                if show_patches:
                    curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners))
                    curr_pts = util.dehomogenize(curr_hom_mat * std_pts_hm)
                    curr_pixel_vals = getPixVals(curr_pts, curr_img_gs)
                    curr_patch = np.reshape(curr_pixel_vals, (res_y, res_x)).astype(np.uint8)
                    curr_patch_resized = cv2.resize(curr_patch, (patch_size_x, patch_size_y))
                if border_size > 0:
                    drawn_img = cv2.copyMakeBorder(curr_img, border_size, border_size, border_size, border_size,
                                                   cv2.BORDER_CONSTANT, border_col)
                    drawn_corners = curr_corners + border_size
                else:
                    drawn_img = curr_img.copy()
                    drawn_corners = curr_corners

                fps_text = 'frame: {:4d}'.format(frame_id)
                if show_error:
                    error = math.sqrt(np.sum(np.square(curr_pixel_vals - init_pixel_vals)))
                    fps_text = '{:s} error: {:f}'.format(fps_text, error)

                cv2.putText(drawn_img, fps_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, text_col)
                drawRegion(drawn_img, drawn_corners, gt_col, 1)
            except IndexError:
                print 'curr_corners: \n', curr_corners
                print 'curr_hom_mat: \n', curr_hom_mat
                print 'Invalid homography or out of range corners'
                continue
        # if img_resize_factor > 1:
        # drawn_img = cv2.resize(drawn_img, resized_img_shape)

        cv2.imshow(window_name, drawn_img)
        if show_patches:
            patch_img = stackImages([templ_patch_resized, curr_patch_resized], stack_order)
            cv2.imshow(curr_patch_win_name, patch_img)
        if show_resized_img:
            drawn_img_resized = cv2.resize(curr_img, resized_img_shape)
            if border_size > 0:
                resized_border = border_size * img_resize_factor
                drawn_img_resized = cv2.copyMakeBorder(drawn_img_resized, resized_border, resized_border,
                                                          resized_border, resized_border,
                                                          cv2.BORDER_CONSTANT, border_col)
                drawn_corners_resized = (curr_corners + border_size) * img_resize_factor
            else:
                drawn_corners_resized = curr_corners * img_resize_factor

            drawRegion(drawn_img_resized, drawn_corners_resized, gt_col, 1)
            cv2.putText(drawn_img_resized, fps_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, text_col)
            cv2.imshow(resized_window_name, drawn_img_resized)

    corrected_gt.append(curr_corners)
    n_corrected_gt = len(corrected_gt)
    print 'Saving corrected GT to {:s}'.format(corrected_corners_fname)
    out_file = open(corrected_corners_fname, 'w')
    out_file.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
    # write corrected gt
    for i in xrange(n_corrected_gt):
        writeCorners(out_file, corrected_gt[i], i + 1)
    # write original gt for frames without corrected gt
    for i in xrange(n_corrected_gt, ground_truth.shape[0]):
        gt_corners = np.asarray([ground_truth[i, 0:2].tolist(),
                                 ground_truth[i, 2:4].tolist(),
                                 ground_truth[i, 4:6].tolist(),
                                 ground_truth[i, 6:8].tolist()]).T
        writeCorners(out_file, gt_corners, i + 1)
    out_file.close()














