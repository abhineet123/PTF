import os
import sys
import shutil
import time
import cv2
import numpy as np
from pprint import pprint
from skimage import feature, filters
from PIL import Image, ImageDraw

import paramparse

from Misc import col_rgb, add_suffix

cols = {
    1: 'white',
    2: 'red',
    3: 'green',
    4: 'blue',
    5: 'magenta',
    6: 'cyan',
    7: 'yellow',
    8: 'forest_green',
    9: 'orange',
    10: 'purple',
}

mask_help = {
    'left_button+drag / ctrl+shift+drag': 'draw mask boundary',
    'shift+left_button+drag': 'delete mask within the given radius of the pointer',
    'shift+drag': 'show the region within which mask would be deleted when left_button is pressed',
    'ctrl+left_button': 'add a single point to the mask',
    'ctrl+drag': 'show where and how next point would be added if left_button is clicked',
    'alt+left_button+drag / arrow keys': 'move the entire mask around',
    'ctrl+right_button / a': 'run augmentation without background selection dialogue (unless necessary)',
    'alt+right_button / A': 'run augmentation with background selection dialogue',
    'ctrl+shift+right_button  / c': 'clean mask points to generate a single contour',
    'shift+right_button': 'delete all mask points',
    'right_button / b': 'start paint mode',
    'middle_button / enter': 'exit, clean mask points to generate a single contour '
                             'and apply changes including bounding box',
    'alt+middle_button / ctrl+enter': 'exit, clean mask points to generate a single contour '
                                      'and apply changes excluding bounding box',
    'shift+middle_button / q': 'exit and discard changes',
    'ctrl+shift+middle_button / esc': 'exit and apply changes',
    'wheel': 'change the drawing window size',
    'shift+wheel / +, - / >, <': 'change the mask deletion radius',
    'ctrl+shift+wheel': 'change the magnified patch area without changing the magnified window size',
    'ctrl+alt+shift+wheel': 'change the magnified window size without changing the magnified patch area',
    'm': 'toggle the magnified window visibility',
}


def runHED(shape_patch, hed_net):
    print('Running HED...')

    hed_start_t = time.time()
    (H, W) = shape_patch.shape[:2]
    blob = cv2.dnn.blobFromImage(shape_patch, scalefactor=1.0, size=(W, H),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)
    hed_net.setInput(blob)
    hed_img = hed_net.forward()
    hed_img = cv2.resize(hed_img[0, 0], (W, H))
    hed_img = (255 * hed_img).astype("uint8")
    hed_end_t = time.time()
    print('time taken: {} secs'.format(hed_end_t - hed_start_t))
    cv2.imshow('hed_img', hed_img)

    threshold = 50
    hed_mask = hed_pts = None

    def update_threshold(x):
        nonlocal threshold, hed_mask, hed_pts
        threshold = x
        _, hed_binary = cv2.threshold(hed_img, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('hed_binary', hed_binary)

        hed_pts, _ = contourPtsFromMask(hed_binary)
        hed_mask, _ = contourPtsToMask(hed_pts, shape_patch)
        cv2.imshow('hed_mask', hed_mask)

    update_threshold(threshold)
    cv2.createTrackbar('threshold', 'hed_binary', threshold, 255, update_threshold)

    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyWindow('hed_img')
        cv2.destroyWindow('hed_binary')
        cv2.destroyWindow('hed_mask')
        return None

    cv2.destroyWindow('hed_img')
    cv2.destroyWindow('hed_binary')
    cv2.destroyWindow('hed_mask')

    return hed_mask


def getContourPts(mask_pts=None, shape=None, show_img=0, verbose=1,
                  patch_img=None, mag_patch_size=50, mag_win_size=800, show_magnified_window=1):
    if shape is None:
        show_img = 0
    else:
        height, width = shape

    n_cols = len(cols)

    n_pts = len(mask_pts)
    mask_segments = []
    curr_segment = []

    for pt_id in range(n_pts):
        _pt = mask_pts[pt_id]
        if not curr_segment:
            if _pt[2]:
                curr_segment.append(pt_id)
            continue
        if not _pt[2]:
            curr_segment.append(pt_id)
            if curr_segment[1] - curr_segment[0] > 2:
                mask_segments.append(curr_segment)
            else:
                if verbose:
                    print('Discarding too short segment {}'.format(curr_segment))
            curr_segment = []
    if len(curr_segment) == 1:
        curr_segment.append(n_pts)
        if curr_segment[1] - curr_segment[0] > 2:
            mask_segments.append(curr_segment)
        else:
            if verbose:
                print('Discarding too short segment {}'.format(curr_segment))

    n_segments = len(mask_segments)
    if n_segments == 0:
        if verbose:
            print('No segments found')
        return []

    if verbose:
        print('Found {} segments: {}'.format(n_segments, mask_segments))

    end_pts_to_segment_dict = {mask_segments[i][0]: (i, 0) for i in range(n_segments)}
    end_pts_to_segment_dict.update({mask_segments[i][1] - 1: (i, 1) for i in range(n_segments)})

    # print('end_pts_to_segment_dict: {}'.format(end_pts_to_segment_dict))

    free_end_pts = list(end_pts_to_segment_dict.keys())

    # print('free_end_pts: {}'.format(free_end_pts))

    contour_pts = [(x, y, 1) for x, y, _ in mask_pts[mask_segments[0][0]:mask_segments[0][1]]]
    free_end_pts.remove(mask_segments[0][0])
    free_end_pts.remove(mask_segments[0][1] - 1)
    # to_remove = mask_segments[0][1] - 1
    min_seg_id = 0
    _id = 0
    while True:
        # try:
        #     free_end_pts.remove(to_remove)
        # except ValueError as e:
        #     print('{} not in free_end_pts'.format(to_remove))
        #     # raise ValueError(e)

        if not free_end_pts:
            break

        # print('free_end_pts: {}'.format(free_end_pts))
        # print('end_pts_to_segment_dict: {}'.format(end_pts_to_segment_dict))

        x, y, _ = contour_pts[-1]
        # x, y, _ = mask_pts[to_remove]

        curr_dists = [abs(x - mask_pts[k][0]) + abs(y - mask_pts[k][1]) for k in free_end_pts]
        min_dist = min(curr_dists)

        _min_id = curr_dists.index(min_dist)
        min_end_pt = free_end_pts[_min_id]

        min_id = end_pts_to_segment_dict[min_end_pt]

        min_seg_id = min_id[0]
        rev_pts = min_id[1]

        # print('_min_id: {} min_end_pt: {} min_id: {}'.format(_min_id, min_end_pt, min_id))

        # if min_id != i:
        #     mask_segments[min_id], mask_segments[i] = mask_segments[i], mask_segments[min_id]
        col_id = (_id % n_cols) + 1
        _id += 1

        curr_pts = [(x, y, col_id) for x, y, _ in
                    mask_pts[mask_segments[min_seg_id][0]:mask_segments[min_seg_id][1]]]
        if rev_pts:
            curr_pts = reversed(curr_pts)
            # free_end_pts.remove(mask_segments[min_seg_id][1] - 1)
            # to_remove = mask_segments[min_seg_id][0]
        else:
            pass
            # free_end_pts.remove(mask_segments[min_seg_id][0])
            # to_remove = mask_segments[min_seg_id][1] - 1

        contour_pts += curr_pts

        # print('to_remove: {}'.format(to_remove))
        # print('min_seg_id: {}'.format(min_seg_id))

        free_end_pts.remove(mask_segments[min_seg_id][0])
        free_end_pts.remove(mask_segments[min_seg_id][1] - 1)

    # contour_pts += [(x, y, col_id) for x, y, _ in mask_pts[mask_segments[-1][0]:mask_segments[-1][1]]]

    # close the contour
    if contour_pts[-1] != contour_pts[0]:
        contour_pts.append(contour_pts[0])

    n_contour_pts = len(contour_pts)
    if verbose:
        print('Found {} contour_pts'.format(n_contour_pts))

    if show_img:
        bin_img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(n_contour_pts - 1):
            pt1, pt2 = contour_pts[i], contour_pts[i + 1]
            col_id = pt2[2]
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))

            col = col_rgb[cols[col_id]]
            # col = (255, 255, 255)

            bin_img = cv2.line(bin_img, pt1, pt2, col, thickness=2)

        cv2.imshow('mask segments', bin_img)

    # bin_img_gs = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    # im2, contours, hierarchy = cv2.findContours(bin_img_gs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)

    # bin_img2 = np.zeros((height, width, 3), dtype=np.uint8)
    # cv2.drawContours(bin_img2, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # cv2.imshow('bin_img2', bin_img2)

    contour_pts = [[x, y] for x, y, _ in contour_pts]

    return contour_pts


def contourPtsToMask(contour_pts, patch_img, col=(255, 255, 255), blend_ratio=0.5):
    # np.savetxt('contourPtsToMask_mask_pts.txt', contour_pts, fmt='%.6f')

    mask_img = np.zeros_like(patch_img, dtype=np.uint8)
    # if not isinstance(contour_pts, list):
    #     raise SystemError('contour_pts must be a list rather than {}'.format(type(contour_pts)))
    if len(contour_pts) > 0:
        mask_img = cv2.fillPoly(mask_img, np.array([contour_pts, ], dtype=np.int32), col)
    if blend_ratio >= 0:
        blended_img = np.array(Image.blend(Image.fromarray(patch_img), Image.fromarray(mask_img), blend_ratio))

        return mask_img, blended_img

    return mask_img


def contourPtsFromMask(mask_img):
    # print('Getting contour pts from mask...')
    if len(mask_img.shape) == 3:
        mask_img_gs = np.squeeze(mask_img[:, :, 0]).copy()
    else:
        mask_img_gs = mask_img.copy()

    ret = cv2.findContours(mask_img_gs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    _contour_pts, _ = ret
    if not _contour_pts:
        return [], []
    contour_pts = list(np.squeeze(_contour_pts[0]))

    n_contours = len(_contour_pts)
    # print('n_contours: {}'.format(n_contours))
    # print('_contour_pts: {}'.format(_contour_pts))
    # print('contour_pts: {}'.format(type(contour_pts)))

    if n_contours > 1:
        max_len = len(contour_pts)
        for _pts in _contour_pts[1:]:
            # print('_pts: {}'.format(_pts))
            _pts = np.squeeze(_pts)
            _len = len(_pts)
            if max_len < _len:
                contour_pts = _pts
                max_len = _len

    # print('contour_pts len: {}'.format(len(contour_pts)))
    mask_pts = [[x, y, 1] for x, y in contour_pts]

    return contour_pts, mask_pts


def addMask(in_img, params, augment=None, hed_net=None):
    """

    :param in_img:
    :param MaskParams params:
    :param augment:
    :param hed_net:
    :return:
    """
    disp_size = params.disp_size
    # border_size = params.border_size
    del_thresh = params.del_thresh
    show_magnified_window = params.show_magnified_window
    mag_patch_size = params.mag_patch_size
    mag_win_size = params.mag_win_size
    # mag_thresh_t = params.mag_thresh_t
    show_pts = params.show_pts
    show_img = params.show_img

    xmin, ymin = 0, 0

    h, w = in_img.shape[:2]
    scale_x, scale_y = disp_size[0] / w, disp_size[1] / h
    scale_factor = min(scale_x, scale_y)
    shape_patch = cv2.resize(in_img, (0, 0), fx=scale_factor, fy=scale_factor)

    print('disp_size: ', disp_size)
    print('scale_factor: ', scale_factor)
    print('shape_patch: ', shape_patch.shape)

    mask_pts = []

    sel_pt_id = -1
    is_continuous = 0
    _exit_mask = 0
    no_save = 0
    draw_mask = 0
    start_painting_mode = 0
    paint_mode = 0
    discard_changes = 0
    clean_mask_pts = 0
    prev_mouse_pt = []
    prev_rect_pts = []
    mouse_x = mouse_y = 0
    end_pts = []
    start_id = 0
    mag_prev_t = 0
    blended_img = mask_img = disp_img = None

    max_dist = del_thresh * del_thresh

    mouse_whl_keys_to_flags = {
        'none': (7864320, -7864320),
        'ctrl': (7864328, -7864312),
        'alt': (7864352, -7864288),
        'shift': (7864336, -7864304),
        'ctrl+alt': (7864360, -7864280),
        'ctrl+shift': (7864344, -7864296),
        'alt+shift': (7864368, -7864272),
        'ctrl+alt+shift': (7864376, -7864264),
    }

    def drawContour(img, mask_pts, curr_pt=None,
                    cursor_thickness=0, show_centroid=0,
                    start_id=0, show_pts=0, contour_img=None):
        nonlocal end_pts

        n_pts = len(mask_pts)

        if contour_img is None:
            if start_id == 0:
                _img = np.copy(img)
            else:
                _img = img
            # print('mask_pts: {}'.format(mask_pts))
            pt1 = pt2 = None
            # pts = []
            end_pts = []
            _is_continuous = 0
            # min_dist = np.inf
            # _nearest_end_pt = mask_pts[0]
            for i in range(start_id, n_pts - 1):
                pt1, pt2 = mask_pts[i], mask_pts[i + 1]
                if not pt1[2] or not pt2[2]:
                    if _is_continuous:
                        end_pts.append(pt1)
                        # if curr_pt:
                        #     dist = abs(curr_pt[0] - pt1[0]) + abs(curr_pt[1] - pt1[1])
                        #     if dist < min_dist:
                        #         min_dist = dist
                        #         _nearest_end_pt = pt1
                    _is_continuous = 0
                    continue

                # pts.append(pt1[:2])
                # pts.append(pt2[:2])

                _img = cv2.line(_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                                (0, 255, 0), thickness=2)
                if show_pts:
                    _img = cv2.drawMarker(_img, (int(pt1[0]), int(pt1[1])), (0, 255, 0), cv2.MARKER_STAR,
                                          markerSize=10)

                if not _is_continuous:
                    _is_continuous = 1
                    end_pts.append(pt1)
                    # if curr_pt:
                    #     dist = abs(curr_pt[0] - pt1[0]) + abs(curr_pt[1] - pt1[1])
                    #     if dist < min_dist:
                    #         min_dist = dist
                    #         _nearest_end_pt = pt1
                    # end_ptsa.append(pt1)
            contour_img = np.copy(_img)
            if curr_pt is not None:
                if _is_continuous:
                    end_pts.append(pt2)

                curr_pt = (int(curr_pt[0]), int(curr_pt[1]))
                if cursor_thickness > 0:
                    cv2.circle(_img, curr_pt, cursor_thickness, (0, 0, 255), 1)
                else:
                    if n_pts == 0:
                        _img = cv2.drawMarker(_img, curr_pt, (0, 255, 0),
                                              cv2.MARKER_STAR, markerSize=10)
                    elif n_pts == 1 and mask_pts[0][2]:
                        prev_pt = (int(mask_pts[0][0]), int(mask_pts[0][1]))
                        _img = cv2.line(_img, prev_pt, curr_pt,
                                        (0, 255, 0), thickness=2)
                    elif end_pts:
                        if len(end_pts) == 1:
                            _nearest_end_pt = end_pts[0]
                        else:
                            dists = [abs(curr_pt[0] - pt[0]) + abs(curr_pt[1] - pt[1]) for pt in end_pts]
                            _nearest_end_pt = end_pts[dists.index(min(dists))]
                        prev_pt = (int(_nearest_end_pt[0]), int(_nearest_end_pt[1]))
                        _img = cv2.line(_img, prev_pt, curr_pt, (0, 255, 0), thickness=2)
            if show_centroid:
                _h, _w = _img.shape[:2]
                mask_centroid = [int(_w / 2), int(_h / 2)]
                # mask_centroid = np.mean(mask_pts, axis=0).astype(np.int32)
                cv2.circle(_img, (mask_centroid[0], mask_centroid[1]),
                           5, (0, 255, 0), -1)

            if _is_continuous:
                if show_pts:
                    _img = cv2.drawMarker(_img, (int(pt2[0]), int(pt2[1])), (0, 255, 0),
                                          cv2.MARKER_STAR, markerSize=10)
        else:
            _img = np.copy(contour_img)
            if cursor_thickness > 0:
                # pass

                # x, y = curr_pt
                # img_h, img_w = _img.shape[:2]
                # min_x, min_y = max(0, x - cursor_thickness), max(0, y - cursor_thickness)
                # max_x, max_y = min(img_w, x + cursor_thickness), min(img_h, y + cursor_thickness)
                # _img[min_y:max_y, min_x:max_x, :] += 10

                cv2.circle(_img, curr_pt, cursor_thickness, (0, 0, 255), 1)

        return _img, contour_img

    _shape_patch, _contour_patch = drawContour(shape_patch, mask_pts, show_pts=show_pts)

    def showMagnifiedWindow(x, y, source_patch, draw_marker=1,
                            win_name='Magnified', marker_col=(0, 0, 255)):
        _h, _w = _shape_patch.shape[:2]
        min_x, max_x = max(0, x - mag_patch_size), min(_w - 1, x + mag_patch_size)
        min_y, max_y = max(0, y - mag_patch_size), min(_h - 1, y + mag_patch_size)

        _x, _y = x - min_x, y - min_y

        mag_patch = np.copy(source_patch[min_y:max_y, min_x:max_x, :])
        if draw_marker == 1:
            mag_patch = cv2.circle(mag_patch, (_x, _y), 1, marker_col, -1)
        elif draw_marker == 2:
            _min_x, _min_y = _x - del_thresh, _y - del_thresh
            _max_x, _max_y = _x + del_thresh, _y + del_thresh
            # mag_patch[_min_y:_max_y, _min_x:_max_x, :] += 10
            cv2.rectangle(mag_patch, (_min_x, _min_y), (_max_x, _max_y), marker_col, 1)

        __h, __w = mag_patch.shape[:2]
        border_x, border_y = int(mag_patch_size - __w / 2), int(mag_patch_size - __h / 2)
        if border_x or border_y:
            mag_patch = cv2.copyMakeBorder(mag_patch, border_y, border_y, border_x, border_x,
                                           cv2.BORDER_CONSTANT)

        mag_patch = cv2.resize(mag_patch, (mag_win_size, mag_win_size))

        cv2.imshow(win_name, mag_patch)

        # mag_prev_t = time.time()

    def paintMouseHandler(event, x, y, flags=None, param=None):
        nonlocal mask_pts, contour_pts, blended_img, mask_img, disp_img, del_thresh, mag_patch_size, mag_win_size, \
            mouse_x, mouse_y, prev_mouse_pt, draw_mask_kb, paint_mode

        # def getNearestPts(x, y, max_dist, max_dist_sqr):
        #     min_x, min_y = x - max_dist, y - max_dist
        #     max_x, max_y = x + max_dist, y + max_dist
        #     valid_pts = [(_x, _y) for _x in range(min_x, max_x) for _y in range(min_y, max_y)
        #                  # if ((x - _x) ** 2 + (y - _y) ** 2) <= max_dist_sqr
        #                  ]
        #     return valid_pts

        paint_mode = 1
        mouse_x, mouse_y = x, y
        # refresh_paint_win = 1
        draw_marker = 1
        marker_col = (0, 255, 0)
        # _show_magnified_window = 0

        if event == cv2.EVENT_MOUSEMOVE:
            # print('flags: {}'.format(flags))
            # refresh_paint_win = 1
            if flags == 1 or flags == 25:
                # left button
                min_x, min_y = x - del_thresh, y - del_thresh
                max_x, max_y = x + del_thresh, y + del_thresh
                mask_img[min_y:max_y, min_x:max_x, :] = 255
                blended_img[min_y:max_y, min_x:max_x, :] = (shape_patch[min_y:max_y, min_x:max_x,
                                                            :] + 255.0) / 2.0
                marker_col = (0, 255, 0)
                # _show_magnified_window = 1
                # pts = getNearestPts(x, y, max_dist, max_dist_sqr)
                # for x, y in pts:
                #     mask_img[y, x, :] = 255
                #     blended_img[y, x, 0] = int((255+patch_img[y, x, 0])/2)
                #     blended_img[y, x, 1] = int((255+patch_img[y, x, 1])/2)
                #     blended_img[y, x, 2] = int((255+patch_img[y, x, 2])/2)
            elif flags == 17:
                # shift + left button
                min_x, min_y = x - del_thresh, y - del_thresh
                max_x, max_y = x + del_thresh, y + del_thresh
                mask_img[min_y:max_y, min_x:max_x, :] = 0
                blended_img[min_y:max_y, min_x:max_x, :] = (shape_patch[min_y:max_y, min_x:max_x, :]) / 2.0
                # _show_magnified_window = 1

                marker_col = (0, 0, 255)

                # pts = getNearestPts(x, y, max_dist, max_dist_sqr)
                # for x, y in pts:
                #     mask_img[y, x, :] = 0
                #     blended_img[y, x, 0] = int((0+patch_img[y, x, 0])/2)
                #     blended_img[y, x, 1] = int((0+patch_img[y, x, 1])/2)
                #     blended_img[y, x, 2] = int((0+patch_img[y, x, 2])/2)
            elif flags == 32:
                draw_marker = 0
            elif flags == 16:
                marker_col = (0, 0, 255)
            elif flags == 8:
                draw_marker = 0
            # else:
            #     draw_marker = 0
        elif event == cv2.EVENT_LBUTTONDOWN:
            # print('flags: {}'.format(flags))
            pass
        elif event == cv2.EVENT_RBUTTONUP:
            # print('flags: {}'.format(flags))
            pass
        elif event == cv2.EVENT_MBUTTONDOWN:
            contour_pts, mask_pts = contourPtsFromMask(mask_img)
            draw_mask_kb = 1
            # print('flags: {}'.format(flags))
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                if flags == mouse_whl_keys_to_flags['ctrl+alt+shift'][0]:
                    mag_win_size += 10
                    print('magnified window size increased to {}'.format(mag_win_size))
                elif flags == mouse_whl_keys_to_flags['ctrl+shift'][0]:
                    mag_patch_size -= 1
                    if mag_patch_size < 5:
                        mag_patch_size = 5
                # elif flags == mouse_whl_keys_to_flags['shift'][0]:
                #     pass
                else:
                    if del_thresh < 10:
                        del_thresh += 1
                    else:
                        del_thresh += 5
                    print('del_thresh increased to {}'.format(del_thresh))
            else:
                if flags == mouse_whl_keys_to_flags['ctrl+alt+shift'][1]:
                    mag_win_size -= 10
                    if mag_win_size < 100:
                        mag_win_size = 100
                    print('magnified window size decreased to {}'.format(mag_win_size))

                elif flags == mouse_whl_keys_to_flags['ctrl+shift'][1]:
                    mag_patch_size += 1
                # elif flags == mouse_whl_keys_to_flags['shift'][1]:
                #     pass
                else:
                    if del_thresh < 10:
                        del_thresh = max(del_thresh - 1, 1)
                    else:
                        del_thresh -= 5
                    print('del_thresh decreased to {}'.format(del_thresh))

        # if disp_img is None or not prev_mouse_pt:
        #     disp_img = np.copy(blended_img)
        # else:
        #     _x, _y = prev_mouse_pt
        #     min_x, min_y = _x - del_thresh, _y - del_thresh
        #     max_x, max_y = _x + del_thresh, _y + del_thresh
        #     disp_img[min_y:max_y, min_x:max_x, :] = blended_img[min_y:max_y, min_x:max_x, :]
        # prev_rect_pts = [()]

        if draw_marker:
            disp_img = np.copy(blended_img)
            # disp_img = blended_img
            min_x, min_y = x - del_thresh, y - del_thresh
            max_x, max_y = x + del_thresh, y + del_thresh
            # disp_img = Image.fromarray(disp_img)
            # ImageDraw.Draw(disp_img).rectangle(
            #     [min_x, min_y, max_x, max_y], outline=marker_col)
            # disp_img = np.asarray(disp_img, dtype=np.uint8)

            # disp_img = cv2.drawMarker(disp_img, (x, y), (0, 0, 255),
            #                           cv2.MARKER_SQUARE, markerSize=del_thresh)
            # disp_img[min_y:max_y, min_x:max_x, :] += 10
            cv2.rectangle(disp_img, (min_x, min_y), (max_x, max_y), marker_col, 1)
        else:
            disp_img = blended_img
        cv2.imshow(paint_win_name, disp_img)

        # disp_img = cv2.drawMarker(disp_img, (x, y), (0, 0, 255),
        #                           cv2.MARKER_STAR, markerSize=5)

        # else:
        #     disp_img = blended_img

        # cv2.rectangle(_shape_patch, (min_x, min_y), (max_x, max_y), marker_col, 1)

        if show_magnified_window:
            showMagnifiedWindow(x, y, _shape_patch, draw_marker=2,
                                marker_col=marker_col,
                                # win_name='Paint Magnified'
                                )
        # cv2.imshow('binary mask', mask_img)
        prev_mouse_pt = (x, y)

    def drawMouseHandler(event, x, y, flags=None, param=None):
        nonlocal mask_pts, sel_pt_id, disp_size, shape_patch, scale_factor, prev_mouse_pt, mouse_x, mouse_y, \
            is_continuous, _exit_mask, del_thresh, draw_mask, max_dist, clean_mask_pts, discard_changes, \
            _shape_patch, mag_patch_size, mag_win_size, start_id, start_painting_mode, cursor_thickness, \
            show_centroid, \
            blended_img, mask_img, paint_mode, _contour_patch

        paint_mode = 0

        _h, _w = shape_patch.shape[:2]
        x = max(min(x, _w - 1), 0)
        y = max(min(y, _h - 1), 0)

        mouse_x, mouse_y = x, y

        if draw_mask > 1:
            draw_mask = 1
        else:
            draw_mask = 0

        if sel_pt_id >= len(mask_pts):
            sel_pt_id = -1

        single_pt_mode = 0
        start_id = 0

        def getNearestPts(pts, pt, max_dist):
            if not pts:
                return []
            n_pts = len(pts)
            x, y = pt
            dist = [(x - _x) ** 2 + (y - _y) ** 2 if _f == 1 else np.inf for _x, _y, _f in pts]
            valid_pt_ids = [i for i in range(n_pts) if dist[i] < max_dist]

            return valid_pt_ids

        # print('flags: {}'.format(flags))

        continuity_broken = 1
        if event == cv2.EVENT_MOUSEMOVE:
            # print('flags: {}'.format(flags))
            if flags == 1 or flags == 25 or flags == 24:
                continuity_broken = 0
                if not is_continuous:
                    is_continuous = 1
                    mask_pts.append([0, 0, 0])
                    mask_pts.append([x, y, 1])
                else:
                    mask_pts.append([x, y, 1])
                    start_id = len(mask_pts) - 2
                draw_mask = 1
                is_continuous = 1
            elif flags == 17:
                # shift
                pt_ids = getNearestPts(mask_pts, (x, y), max_dist)
                for _id in pt_ids:
                    mask_pts[_id][2] = 0
                draw_mask = 5
            elif flags == 33:
                # alt + left mouse
                if prev_mouse_pt:
                    tx, ty = x - prev_mouse_pt[0], y - prev_mouse_pt[1]
                    mask_pts = [[_x + tx, _y + ty, f] for _x, _y, f in mask_pts]
                prev_mouse_pt = [x, y]
                draw_mask = 1
                single_pt_mode = 1
            elif flags == 16:
                # delete_mode = 1
                draw_mask = 3
            elif flags == 8:
                draw_mask = 2
        elif event == cv2.EVENT_LBUTTONDOWN:
            # print('flags: {}'.format(flags))
            if flags == 9:
                if end_pts:
                    dists = [abs(x - pt[0]) + abs(y - pt[1]) for pt in end_pts]
                    _nearest_end_pt = end_pts[dists.index(min(dists))]
                    if _nearest_end_pt != mask_pts[-1]:
                        mask_pts.append([0, 0, 0])
                        mask_pts.append(_nearest_end_pt)
                mask_pts.append([x, y, 1])
        elif event == cv2.EVENT_RBUTTONUP:
            print('flags: {}'.format(flags))
            if flags == 24:
                # ctrl + shift
                _contour_pts = getContourPts(mask_pts)
                print('Cleaning up the mask points...')
                mask_pts = [[x, y, 1] for x, y in _contour_pts]
                draw_mask = 1
            elif flags == 16:
                # shift
                print('Deleting all mask points ...')
                mask_pts = []
                draw_mask = 1
            elif flags == 8:
                # ctrl
                run_augmentation(use_prev=1)
            elif flags == 32:
                # alt
                run_augmentation(use_prev=0)
            else:
                # if not delete_mode:
                _contour_pts = getContourPts(mask_pts, shape_patch.shape[:2], show_img=1)
                mask_img, blended_img = contourPtsToMask(_contour_pts, shape_patch)
                start_painting_mode = 1
        elif event == cv2.EVENT_MBUTTONDOWN:
            print('flags: {}'.format(flags))
            if flags == 4:
                clean_mask_pts = 2
                _exit_mask = 1
            if flags == 36:
                clean_mask_pts = 1
                _exit_mask = 1
            elif flags == 28:
                _exit_mask = 1
            elif flags == 20:
                discard_changes = 1
                _exit_mask = 1
        elif event == cv2.EVENT_MOUSEWHEEL:
            # flags -= 7864320
            print('flags: {}'.format(flags))
            _disp_size = disp_size
            if flags > 0:
                if flags == mouse_whl_keys_to_flags['ctrl+alt+shift'][0]:
                    mag_win_size += 10
                    print('magnified window size increased to {}'.format(mag_win_size))
                elif flags == mouse_whl_keys_to_flags['ctrl+shift'][0]:
                    mag_patch_size -= 1
                    if mag_patch_size < 5:
                        mag_patch_size = 5
                    # show_magnified_window = 1
                elif flags == mouse_whl_keys_to_flags['shift'][0]:
                    if del_thresh < 10:
                        del_thresh += 1
                    else:
                        del_thresh += 5
                    print('del_thresh increased to {}'.format(del_thresh))
                    max_dist = del_thresh * del_thresh
                    # delete_mode = 1
                    draw_mask = 3
                elif flags == mouse_whl_keys_to_flags['none'][0]:
                    _disp_size = tuple([min(2000, x + 10) for x in disp_size])
            else:
                if flags == mouse_whl_keys_to_flags['ctrl+alt+shift'][1]:
                    mag_win_size -= 10
                    if mag_win_size < 100:
                        mag_win_size = 100
                    print('magnified window size decreased to {}'.format(mag_win_size))

                elif flags == mouse_whl_keys_to_flags['ctrl+shift'][1]:
                    mag_patch_size += 1
                    # show_magnified_window = 1
                elif flags == mouse_whl_keys_to_flags['shift'][1]:
                    if del_thresh > 10:
                        del_thresh -= 5
                    else:
                        del_thresh -= 1
                    if del_thresh < 1:
                        del_thresh = 1
                    max_dist = del_thresh * del_thresh
                    print('del_thresh decreased to {}'.format(del_thresh))
                    # delete_mode = 1
                    draw_mask = 3
                elif flags == mouse_whl_keys_to_flags['none'][1]:
                    _disp_size = tuple([max(200, x - 10) for x in disp_size])
            if disp_size != _disp_size:
                scale_x, scale_y = _disp_size[0] / w, _disp_size[1] / h
                _scale_factor = min(scale_x, scale_y)
                shape_patch = cv2.resize(in_img, (0, 0), fx=_scale_factor, fy=_scale_factor)
                print('_disp_size: ', _disp_size)
                print('_scale_factor: ', _scale_factor)
                print('shape_patch: ', shape_patch.shape)
                k = _scale_factor / scale_factor
                mask_pts = [[x * k, y * k, f] for x, y, f in mask_pts]
                draw_mask = 1
                disp_size = _disp_size
                scale_factor = _scale_factor

            # print('mask_pts: {}'.format(mask_pts))
            # print('mask: {}'.format(mask))

        # k = cv2.waitKey(1)
        # print('k: {}'.format(k))
        # if k == 27:
        #     cv2.destroyWindow(draw_win_name)
        #     return
        if not single_pt_mode:
            prev_mouse_pt = []

        if continuity_broken:
            is_continuous = 0

        if draw_mask:
            contour_patch = None
            curr_pt = None
            cursor_thickness = 0
            show_centroid = 0
            if draw_mask == 2:
                curr_pt = (x, y)
            elif draw_mask == 3:
                curr_pt = (x, y)
                cursor_thickness = del_thresh
                contour_patch = _contour_patch
            elif draw_mask == 4:
                show_centroid = 1
            elif draw_mask == 5:
                curr_pt = (x, y)
                cursor_thickness = del_thresh
            if start_id > 0:
                _img = _shape_patch
            else:
                _img = shape_patch
            _shape_patch, _contour_patch = drawContour(
                _img, mask_pts, curr_pt, cursor_thickness, show_centroid,
                start_id, show_pts, contour_img=contour_patch)
            cv2.imshow(draw_win_name, _shape_patch)

        if show_magnified_window:
            showMagnifiedWindow(x, y, _shape_patch)

        if start_painting_mode:
            cv2.imshow(paint_win_name, blended_img)
            cv2.setMouseCallback(paint_win_name, paintMouseHandler)
            start_painting_mode = 0

        # print('continuity_broken: {}'.format(continuity_broken))
        # print('start_id: {}'.format(start_id))

    def run_augmentation(use_prev=1):
        if augment is not None:
            mask = [(xmin + x / scale_factor, ymin + y / scale_factor, f)
                    for (x, y, f) in mask_pts]
            augment(None, mask, use_prev=use_prev, save_seq=0)

    paint_win_name = 'Paint the mask'
    draw_win_name = 'Draw the mask'
    cv2.imshow(draw_win_name, _shape_patch)
    cv2.setMouseCallback(draw_win_name, drawMouseHandler,
                         # param=(mask_pts, shape_patch)
                         )
    draw_mask_kb = 0
    invert_mask = 0
    while not _exit_mask:
        # while True:
        k = cv2.waitKeyEx(100)

        if not draw_mask_kb and k < 0:
            continue

        print('k: {}'.format(k))

        if k == ord('p'):
            show_pts = 1 - show_pts
            draw_mask_kb = 1
        elif k == ord('l'):
            print('Running Laplacian filtering...')
            shape_patch_gs = cv2.cvtColor(shape_patch, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(shape_patch_gs, cv2.CV_64F)
            threshold = 0

            def update_threshold(x):
                nonlocal threshold
                threshold = x
                _, laplacian_binary = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)
                cv2.imshow('laplacian_binary', laplacian_binary)

            _, laplacian_binary = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('laplacian', laplacian)
            cv2.imshow('laplacian_binary', laplacian_binary)

            cv2.createTrackbar('threshold', 'laplacian_binary', int(threshold), 20, update_threshold)

        elif k == ord('s'):
            print('Running Sobel filtering...')
            shape_patch_gs = cv2.cvtColor(shape_patch, cv2.COLOR_BGR2GRAY)
            # shape_patch_gs = cv2.GaussianBlur(shape_patch_gs, (k, k), 0)
            edgeX = cv2.Sobel(shape_patch_gs, cv2.CV_16S, 1, 0)
            edgeY = cv2.Sobel(shape_patch_gs, cv2.CV_16S, 0, 1)

            edgeX = np.uint8(np.absolute(edgeX))
            edgeY = np.uint8(np.absolute(edgeY))
            edge = cv2.bitwise_or(edgeX, edgeY)

            threshold = 50

            def update_threshold(x):
                nonlocal threshold
                threshold = x
                _, sobel_binary = cv2.threshold(edge, threshold, 255, cv2.THRESH_BINARY)
                cv2.imshow('sobel_binary', sobel_binary)

            _, sobel_binary = cv2.threshold(edge, threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('edge', edge)
            cv2.imshow('sobel_binary', sobel_binary)
            cv2.createTrackbar('threshold', 'sobel_binary', threshold, 100, update_threshold)

        elif k == ord('o'):
            print('Running Otsu thresholding...')
            shape_patch_gs = cv2.cvtColor(shape_patch, cv2.COLOR_BGR2GRAY)
            val = filters.threshold_otsu(shape_patch_gs)
            otsu_mask = np.array(shape_patch_gs < val, dtype=np.uint8) * 255
            cv2.imshow('otsu_mask', otsu_mask)
        elif k == ord('E'):
            shape_patch_gs = cv2.cvtColor(shape_patch, cv2.COLOR_BGR2GRAY)

            # shape_patch_gs = ndi.gaussian_filter(shape_patch_gs, 4)
            sigma = 3
            edges_canny = feature.canny(shape_patch_gs, sigma=sigma)
            edges_canny = np.array(edges_canny, dtype=np.uint8) * 255
            cv2.imshow('edges_canny', edges_canny)

            def update_sigma(x):
                nonlocal sigma
                sigma = x
                edges_canny = feature.canny(shape_patch_gs, sigma=sigma)
                edges_canny = np.array(edges_canny, dtype=np.uint8) * 255
                cv2.imshow('edges_canny', edges_canny)

            cv2.createTrackbar('sigma', 'edges_canny', sigma, 100, update_sigma)

        elif k == ord('e'):
            print('Running Canny edge detection...')
            threshold1 = 50
            threshold2 = 25
            canny_edge_patch = cv2.Canny(shape_patch, threshold1, threshold2)
            cv2.imshow('canny_edge_patch', canny_edge_patch)

            def update_threshold1(x):
                nonlocal threshold1
                threshold1 = x
                canny_edge_patch = cv2.Canny(shape_patch, threshold1, threshold2)
                cv2.imshow('canny_edge_patch', canny_edge_patch)

            def update_threshold2(x):
                nonlocal threshold2
                threshold2 = x
                canny_edge_patch = cv2.Canny(shape_patch, threshold1, threshold2)
                cv2.imshow('canny_edge_patch', canny_edge_patch)

            cv2.createTrackbar('threshold1', 'canny_edge_patch', threshold1, 1000, update_threshold1)
            cv2.createTrackbar('threshold2', 'canny_edge_patch', threshold2, 1000, update_threshold2)

        elif k == ord('h'):
            if hed_net is not None:
                hed_mask = runHED(shape_patch, hed_net)
                if hed_mask is not None:
                    _, _mask_pts = contourPtsFromMask(hed_mask)
                    mask_pts = [[x, y, 1] for x, y in _mask_pts]
                    mask = [(xmin + x / scale_factor, ymin + y / scale_factor, f)
                            for (x, y, f) in _mask_pts]
        elif k == ord('H'):
            pprint(mask_help)
        elif k == ord('m'):
            show_magnified_window = 1 - show_magnified_window
            if not show_magnified_window:
                cv2.destroyWindow('Magnified')
        elif k == ord('c'):
            print('Cleaning up the mask points...')
            contour_pts = getContourPts(mask_pts)
            mask_pts = [[x, y, 1] for x, y in contour_pts]
            draw_mask_kb = 1
        elif k == ord('a'):
            run_augmentation(use_prev=1)
        elif k == ord('A'):
            run_augmentation(use_prev=0)
        elif k == ord('b'):
            _contour_pts = getContourPts(mask_pts, shape_patch.shape[:2], show_img=1)
            mask_img, blended_img = contourPtsToMask(_contour_pts, shape_patch)
            start_painting_mode = 1
        elif k == ord('q'):
            discard_changes = 1
            _exit_mask = 1
        elif k == 10:
            _exit_mask = 1
            clean_mask_pts = 1
            invert_mask = 1
        elif k == 13:
            _exit_mask = 1
            clean_mask_pts = 2
        elif k == 27:
            _exit_mask = 1
            discard_changes = 1
        elif k == ord('+') or k == ord('>'):
            del_thresh += 5
            print('del_thresh increased to {}'.format(del_thresh))
            max_dist = del_thresh * del_thresh
        elif k == ord('-') or k == ord('<'):
            del_thresh -= 5
            if del_thresh < 1:
                del_thresh = 1
            print('del_thresh decreased to {}'.format(del_thresh))
            max_dist = del_thresh * del_thresh
        elif k == 2490368:
            # up
            mask_pts = [[_x, _y - 1, f] for _x, _y, f in mask_pts]
            draw_mask_kb = 1
        elif k == 2621440:
            # down
            mask_pts = [[_x, _y + 1, f] for _x, _y, f in mask_pts]
            draw_mask_kb = 1
        elif k == 2555904:
            # right
            mask_pts = [[_x + 1, _y, f] for _x, _y, f in mask_pts]
            draw_mask_kb = 1
        elif k == 2424832:
            # left
            mask_pts = [[_x - 1, _y, f] for _x, _y, f in mask_pts]
            draw_mask_kb = 1

        if start_painting_mode:
            cv2.imshow(paint_win_name, blended_img)
            cv2.setMouseCallback(paint_win_name, paintMouseHandler)
            start_painting_mode = 0

        if draw_mask_kb:
            curr_pt = None
            cursor_thickness = 0
            show_centroid = 0
            if draw_mask_kb == 2:
                curr_pt = (mouse_x, mouse_y)
            elif draw_mask_kb == 3:
                curr_pt = (mouse_x, mouse_y)
                cursor_thickness = del_thresh
            elif draw_mask_kb == 4:
                show_centroid = 1
            if start_id > 0:
                _img = _shape_patch
            else:
                _img = shape_patch
            _shape_patch, _contour_patch = drawContour(_img, mask_pts, curr_pt, cursor_thickness,
                                                       show_centroid, start_id, show_pts)
            cv2.imshow(draw_win_name, _shape_patch)
            draw_mask_kb = 0

    if clean_mask_pts:
        print('Cleaning up the mask points...')
        contour_pts = getContourPts(mask_pts)
        mask_pts = [[x, y, 1] for x, y in contour_pts]
    if not discard_changes:
        if paint_mode:
            contour_pts, mask_pts = contourPtsFromMask(mask_img)

        # mask = [(xmin + x / scale_factor, ymin + y / scale_factor, f)
        #         for (x, y, f) in mask_pts]

        # min_border_x, min_border_y = params.min_box_border

        # if min_border_x > 0 or min_border_x > 0:
        #     min_x, min_y = min_border_x, min_border_y
        #     max_x, max_y = width - min_border_x, height - min_border_y
        #
        #     print('Clamping mask to be between {} and {}'.format(
        #         (min_x, min_y), (max_x, max_y)))
        #
        #     mask = [(min(max_x, max(min_x, x)), min(max_y, max(min_y, y)), f)
        #             for (x, y, f) in mask]

        # if clean_mask_pts == 2 and len(mask) > 0:
        #     print('Fixing bounding box using the updated mask')
        #     mask_arr = np.asarray([(x, y) for x, y, _ in mask])
        #     xmin, ymin = np.min(mask_arr, axis=0)
        #     xmax, ymax = np.max(mask_arr, axis=0)

    else:
        print('Discarding changes ...')
        return None, None

    # cv2.destroyWindow(draw_win_name)
    cv2.destroyAllWindows()

    mask_orig = [(xmin + x / scale_factor, ymin + y / scale_factor)
                 for (x, y, _) in mask_pts]
    mask_arr = np.asarray(mask_orig)
    mask_img_orig = contourPtsToMask(mask_orig, in_img, blend_ratio=-1)

    out_img = np.copy(in_img)
    if invert_mask:
        print('inverting mask')
        mask_img_orig = 255 - mask_img_orig

        _, mask_pts = contourPtsFromMask(mask_img_orig)
        mask_orig = [(x, y) for (x, y, _) in mask_pts]
        mask_arr = np.asarray(mask_orig)

    xmin, ymin = np.min(mask_arr, axis=0)
    xmax, ymax = np.max(mask_arr, axis=0)
    xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

    mask_img_orig_inv = (255 - mask_img_orig).astype(np.bool)

    out_img[mask_img_orig_inv] = 0
    out_img = out_img[ymin:ymax + 1, xmin:xmax + 1, ...]

    mask_img_orig_float = mask_img_orig.astype(np.float32) / 255.0

    # mask_img_gauss_3 = cv2.GaussianBlur(mask_img_orig_float, (3, 3), 0)
    # mask_img_gauss_5 = cv2.GaussianBlur(mask_img_orig_float, (5, 5), 0)
    # mask_img_gauss_7 = cv2.GaussianBlur(mask_img_orig_float, (7, 7), 0)
    # mask_img_gauss_9 = cv2.GaussianBlur(mask_img_orig_float, (9, 9), 0)
    #
    # mask_img_gauss_25 = cv2.GaussianBlur(mask_img_orig_float, (25, 25), 0)
    # mask_img_gauss_50 = cv2.GaussianBlur(mask_img_orig_float, (50, 50), 0)
    # mask_img_gauss_100 = cv2.GaussianBlur(mask_img_orig_float, (100, 100), 0)
    # mask_img_gauss_200 = cv2.GaussianBlur(mask_img_orig_float, (200, 200), 0)
    # mask_img_gauss_300 = cv2.GaussianBlur(mask_img_orig_float, (300, 300), 0)
    #
    # out_img_gauss_3 = (in_img.astype(np.float32) * mask_img_gauss_3).astype(np.uint8)
    # out_img_gauss_5 = (in_img.astype(np.float32) * mask_img_gauss_5).astype(np.uint8)
    # out_img_gauss_7 = (in_img.astype(np.float32) * mask_img_gauss_7).astype(np.uint8)
    # out_img_gauss_9 = (in_img.astype(np.float32) * mask_img_gauss_9).astype(np.uint8)

    def get_gauss_image(k):
        mask_img_gauss = cv2.GaussianBlur(mask_img_orig_float, (k, k), 0)
        out_img_gauss = (in_img.astype(np.float32) * mask_img_gauss).astype(np.uint8)
        out_img_gauss = out_img_gauss[ymin:ymax + 1, xmin:xmax + 1, ...]

        if show_img:
            cv2.imshow(f'out_img_gauss_{k}', out_img_gauss)
        return k, out_img_gauss

    # out_img_gauss_25 = (in_img.astype(np.float32) * mask_img_gauss_25).astype(np.uint8)
    # out_img_gauss_50 = (in_img.astype(np.float32) * mask_img_gauss_50).astype(np.uint8)
    # out_img_gauss_100 = (in_img.astype(np.float32) * mask_img_gauss_100).astype(np.uint8)
    # out_img_gauss_200 = (in_img.astype(np.float32) * mask_img_gauss_200).astype(np.uint8)
    # out_img_gauss_300 = (in_img.astype(np.float32) * mask_img_gauss_300).astype(np.uint8)

    # out_img_gauss_3 = np.array(Image.blend(Image.fromarray(in_img), Image.fromarray(mask_img_gauss_3), 1))
    # out_img_gauss_5 = np.array(Image.blend(Image.fromarray(in_img), Image.fromarray(mask_img_gauss_5), 1))
    # out_img_gauss_7 = np.array(Image.blend(Image.fromarray(in_img), Image.fromarray(mask_img_gauss_7), 1))
    # out_img_gauss_9 = np.array(Image.blend(Image.fromarray(in_img), Image.fromarray(mask_img_gauss_9), 1))

    # cv2.imshow('mask_img', mask_img)

    gauss_imgs = dict(
        [
            # get_gauss_image(25),
            get_gauss_image(51),
            get_gauss_image(101),
            # get_gauss_image(201),
            # get_gauss_image(301),
            # get_gauss_image(401),
        ]
    )
    if show_img:
        cv2.imshow('mask_img_orig', mask_img_orig)
        cv2.imshow('out_img', out_img)
        cv2.waitKey(0)

    # cv2.imshow('out_img_gauss_3', out_img_gauss_3)
    # cv2.imshow('out_img_gauss_5', out_img_gauss_5)
    # cv2.imshow('out_img_gauss_7', out_img_gauss_7)
    # cv2.imshow('out_img_gauss_9', out_img_gauss_9)

    return out_img, gauss_imgs


class MaskParams:
    def __init__(self):
        """

        """
        self.src_file_path = ''
        self.disp_size = (900, 900)
        self.border_size = (10, 10)
        self.min_box_border = (1, 1)
        self.del_thresh = 15
        self.show_magnified_window = 1
        self.mag_patch_size = 50
        self.mag_win_size = 800
        self.mag_thresh_t = 0.05
        self.load_boxes = 0
        self.gen_method = 1
        self.show_binary = 0
        self.show_pts = 0
        self.save_test = 0
        self.save_raw = 0
        self.show_img = 1
        self.hed_model_path = '../hed_cv/hed_model'
        self.help = {
            'disp_size': "Size of the window shown for drawing the mask",
            'border_size': "Size of border around the bounding box to include in the mask window",
            'min_box_border': "minimum border to be left around the bounding box to avoid image boundary "
                              "aligned boxes that csn mess up training",
            'del_thresh': "Distance threshold for deleting the existing mask points",
            'show_magnified_window': "Show magnified window around the cursor location",
            'mag_patch_size': "size of patch around the cursor location shown in the magnified window ",
            'mag_win_size': "magnified window size",
            'mag_thresh_t': "minimum time in seconds between successive updates of the magnifying window window size",
            'gen_method': "method used for generating masks: "
                          "0: normalized AST computed by DLT;"
                          "1: simple translation and scaling",
            'show_binary': "Show binary mask in painting mode",
            'show_pts': "Show individual points",
            'save_test': "save unlabeled objects in a separate test sequence",
            'save_raw': "save rsw labels while saving mask sequences",
            'hed_model_path': "hed_model_path",
        }


def main():
    params = MaskParams()
    paramparse.process(params)

    src_file_path = params.src_file_path

    if not src_file_path:
        try:
            from Tkinter import Tk
        except ImportError:
            from tkinter import Tk
        try:
            src_file_path = Tk().clipboard_get()
        except BaseException as e:
            print('Tk().clipboard_get() failed: {}'.format(e))
            return None, None

    src_file_path = src_file_path.replace(os.sep, "/").replace('"', '')

    assert os.path.exists(src_file_path), "src_file does not exist: {}".format(src_file_path)

    src_file = cv2.imread(src_file_path)

    out_img, gauss_imgs = addMask(src_file, params)

    if out_img is None:
        return None, None

    dst_file_path = add_suffix(src_file_path, "backup")
    shutil.move(src_file_path, dst_file_path)

    cv2.imwrite(src_file_path, out_img)

    for k in gauss_imgs:
        out_file_path = add_suffix(src_file_path, f"gauss_{k}")
        cv2.imwrite(out_file_path, gauss_imgs[k])


if __name__ == '__main__':
    main()
