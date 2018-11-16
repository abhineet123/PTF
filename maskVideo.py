import os
import cv2
import numpy as np
import sys
from Misc import processArguments, drawRegion, getDateTime


def getConvexRegion(img, n_pts=-1, col=(0, 0, 255), title=None, line_thickness=1):
    annotated_img = img.copy()
    temp_img = img.copy()
    if title is None:
        title = 'Select the convex region'
    cv2.namedWindow(title)
    cv2.imshow(title, annotated_img)
    pts = []

    def drawLines(img, hover_pt=None):
        if len(pts) == 0:
            cv2.imshow(title, img)
            return
        for i in xrange(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], col, line_thickness)
        if hover_pt is None:
            return
        cv2.line(img, pts[-1], hover_pt, col, line_thickness)
        if n_pts > 0:
            if len(pts) == n_pts - 1:
                cv2.line(img, pts[0], hover_pt, col, line_thickness)
            elif len(pts) == n_pts:
                return
        cv2.imshow(title, img)

    def mouseHandler(event, x, y, flags=None, param=None):
        if n_pts > 0 and len(pts) >= n_pts:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            temp_img = annotated_img.copy()
            drawLines(temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            pass
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(pts) > 0:
                print 'Removing last point'
                del (pts[-1])
            temp_img = annotated_img.copy()
            drawLines(temp_img)
        elif event == cv2.EVENT_RBUTTONUP:
            pass
        elif event == cv2.EVENT_MBUTTONDOWN:
            pass
        elif event == cv2.EVENT_MOUSEMOVE:
            # if len(pts) == 0:
            # return
            temp_img = annotated_img.copy()
            drawLines(temp_img, (x, y))

    cv2.setMouseCallback(title, mouseHandler, param=[annotated_img, temp_img, pts])
    while True:
        if n_pts > 0 and len(pts) >= n_pts:
            break
        key = cv2.waitKey(1)
        if key == 27:
            if n_pts <= 0:
                break
            sys.exit()
        elif key == 13:
            break
    drawLines(annotated_img, pts[0])
    cv2.waitKey(250)
    cv2.destroyWindow(title)
    return pts


params = {
    'root_dir': 'N:\Datasets',
    'seq_name': 'DJI_0020',
    'vid_fmt': 'mov',
    'show_img': 1,
    'n_frames': 1000,
    'n_pts': -1,
    'n_regions': 1,
    'remove_regions': 1,
    'vis_resize_factor': 1.0,
    'read_from_file': 0,
    'region_fname': 'mask_regions.txt',
    'save_fmt': ('avi', 'XVID', 30)
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)

    root_dir = params['root_dir']
    seq_name = params['seq_name']
    show_img = params['show_img']
    vid_fmt = params['vid_fmt']
    n_frames = params['n_frames']
    n_pts = params['n_pts']
    n_regions = params['n_regions']
    remove_regions = params['remove_regions']
    save_fmt = params['save_fmt']
    read_from_file = params['read_from_file']
    region_fname = params['region_fname']
    vis_resize_factor = params['vis_resize_factor']

    resize_vis_images = vis_resize_factor != 1

    src_fname = root_dir + '/' + seq_name + '.' + vid_fmt
    dst_fname = root_dir + '/' + seq_name + '_masked_{:d}_{:s}.'.format(n_frames, getDateTime()) + save_fmt[0]

    print('Reading video file: {:s}'.format(src_fname))
    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        raise StandardError('The video file ' + src_fname + ' could not be opened')

    ret, frame = cap.read()
    if not ret:
        raise StandardError('First frame could not be read')

    regions = []
    n_rows, n_cols, n_channels = frame.shape
    if remove_regions:
        frame_mask = np.zeros((n_rows, n_cols, n_channels), dtype=np.uint8)
        fill_col = (255, 255, 255)
        op_type = 'remove'
        line_col = (0, 0, 255)
    else:
        frame_mask = np.full((n_rows, n_cols, n_channels), fill_value=255, dtype=np.uint8)
        fill_col = (0, 0, 0)
        op_type = 'retain'
        line_col = (0, 255, 0)

    if read_from_file:
        print('Reading regions from {:s}'.format(region_fname))
        region_fid = open(region_fname, 'r')
    else:
        print('Writing regions to {:s}'.format(region_fname))
        region_fid = open(region_fname, 'w')

    if resize_vis_images:
        disp_frame = cv2.resize(frame, (0, 0), fx=vis_resize_factor, fy=vis_resize_factor)
    else:
        disp_frame = frame.copy()

    for i in range(n_regions):
        if read_from_file:
            line_x = region_fid.readline()
            line_y = region_fid.readline()
            if not line_x or not line_y:
                raise StandardError('Region file ended unexpectedly')
            words_x = line_x.split()
            words_y = line_y.split()
            _n_pts = len(words_x)
            if _n_pts != len(words_y):
                raise StandardError('Invalid formatting found in region file')
            if n_pts > 0 and n_pts != _n_pts:
                raise StandardError(
                    'Invalid no. of points {:d} found in region file for region {:d}'.format(_n_pts, i + 1))
            region = []
            for j in range(_n_pts):
                region.append((float(words_x[j]), float(words_y[j])))
        else:
            win_title = 'Select region {:d} to {:s}'.format(i + 1, op_type)
            if n_pts > 0:
                win_title = '{:s} by clicking {:d} points'.format(win_title, n_pts)
            else:
                win_title = '{:s}. Press Esc when done.'.format(win_title)
            region = getConvexRegion(disp_frame, n_pts=n_pts, line_thickness=2, col=line_col,
                                     title=win_title)
            # cv2.destroyWindow(win_title)
            if resize_vis_images:
                region = [(x / vis_resize_factor, y / vis_resize_factor) for x, y in region]
            for j in range(len(region)):
                region_fid.write('{:f}\t'.format(region[j][0]))
            region_fid.write('\n')
            for j in range(len(region)):
                region_fid.write('{:f}\t'.format(region[j][1]))
            region_fid.write('\n')
        corners = np.array(region, dtype=np.int32)
        cv2.fillConvexPoly(frame_mask, corners, fill_col)
        drawRegion(disp_frame, (corners * vis_resize_factor).transpose(),
                   thickness=2, color=line_col)
        if read_from_file and show_img:
            win_title = 'Regions 1-{:d}. Press any key to continue'.format(i + 1)
            cv2.imshow(win_title, disp_frame)
            cv2.waitKey(0)
            cv2.destroyWindow(win_title)

        regions.append(region)

    region_fid.close()

    # frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_GRAY2RGB)

    if show_img:
        if resize_vis_images:
            disp_frame_mask = cv2.resize(frame_mask, (0, 0), fx=vis_resize_factor, fy=vis_resize_factor)
        else:
            disp_frame_mask = frame_mask
        win_title = 'Remove mask. Press any key to continue'
        cv2.imshow(win_title, disp_frame_mask)
        # cv2.imshow('Original mask. Press any key to continue', frame_mask)
        cv2.waitKey(0)
        cv2.destroyWindow(win_title)

    frame_mask = frame_mask.astype(np.bool)

    frame_size = (int(n_cols), int(n_rows))

    video_writer = cv2.VideoWriter()
    if cv2.__version__.startswith('3'):
        video_writer.open(filename=dst_fname, apiPreference=cv2.CAP_FFMPEG,
                          fourcc=cv2.VideoWriter_fourcc(*save_fmt[1]),
                          fps=int(save_fmt[2]), frameSize=frame_size)
    else:
        video_writer.open(filename=dst_fname, fourcc=cv2.cv.CV_FOURCC(*save_fmt[1]),
                          fps=save_fmt[2], frameSize=frame_size)

    if not video_writer.isOpened():
        print('Video file {:s} could not be opened for writing'.format(dst_fname))
        exit(0)

    print('Writing video file: {:s}'.format(dst_fname))
    frame_id = 0
    while True:
        frame[frame_mask] = 0
        if show_img:
            if resize_vis_images:
                _disp_frame = cv2.resize(frame, (0, 0), fx=vis_resize_factor, fy=vis_resize_factor)
            else:
                _disp_frame = frame
            cv2.imshow('Frame', _disp_frame)
            # cv2.imshow('Original Frame', frame)
            if cv2.waitKey(1) == 27:
                break
        video_writer.write(frame)
        frame_id += 1
        if n_frames > 0 and frame_id >= n_frames:
            break
        ret, frame = cap.read()
        if not ret:
            print('Frame {:d} could not be read'.format(frame_id + 1))
            break
        sys.stdout.write('\rDone {:d}/{:d} frames'.format(
            frame_id + 1, n_frames))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

    video_writer.release()
