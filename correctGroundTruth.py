from Misc import readTrackingData
from Misc import getParamDict
from Misc import readDistGridParams
from Misc import drawRegion

import sys
import cv2
import numpy as np

curr_corners=None
curr_pt_id=0

left_click_location=None
left_click_detected=False

final_location=None
final_location_selected=False

title=None

def drawLines(img, col=(0, 0, 255), hover_pt=None):
    if len(curr_locations) == 0:
        cv2.imshow(title, img)
        return
    for i in xrange(len(curr_locations) - 1):
        cv2.line(img, curr_locations[i], curr_locations[i + 1], col, 1)

def getNearestCorner(pt):
    diff_x=curr_corners[0, 0]-pt[0]
    diff_y=curr_corners[1, 0]-pt[1]

    min_dist=diff_x*diff_x + diff_y*diff_y
    min_dist_id=0
    for i in xrange(1, 4):
        diff_x=curr_corners[0, i]-pt[0]
        diff_y=curr_corners[1, i]-pt[1]

        curr_dist=diff_x*diff_x + diff_y*diff_y
        if curr_dist<min_dist:
            min_dist=curr_dist
            min_dist_id=i
    return min_dist_id


def mouseHandler(event, x, y, flags=None, param=None):
    global left_click_location, left_click_detected
    global final_location, final_location_selected
    global curr_locations, curr_pt_id

    if event == cv2.EVENT_LBUTTONDOWN:
        left_click_location = [x, y]
        left_click_detected=True
        return
    elif event == cv2.EVENT_LBUTTONUP:
        final_location = [x, y]
        final_location_selected=True
        pass
    elif event == cv2.EVENT_RBUTTONDOWN:
        pass
    elif event == cv2.EVENT_RBUTTONUP:
        pass
    elif event == cv2.EVENT_MBUTTONDOWN:
        pass
    elif event == cv2.EVENT_MOUSEMOVE:
        # if len(pts) == 0:
        # return
        curr_locations[curr_pt_id]=[x, y]
        temp_img = annotated_img.copy()
        drawLines(temp_img, (x, y))
        cv2.imshow(title, img)

def getTrackingObject2(img, col=(0, 0, 255), title=None):
    annotated_img = img.copy()
    temp_img = img.copy()
    if title is None:
        title = 'Select the object to track'
    cv2.namedWindow(title)
    cv2.imshow(title, annotated_img)
    pts = []

    def drawLines(img, hover_pt=None):
        if len(pts) == 0:
            cv2.imshow(title, img)
            return
        for i in xrange(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], col, 1)
        if hover_pt is None:
            return
        cv2.line(img, pts[-1], hover_pt, col, 1)
        if len(pts) == 3:
            cv2.line(img, pts[0], hover_pt, col, 1)
        elif len(pts) == 4:
            return
        cv2.imshow(title, img)



    cv2.setMouseCallback(title, mouseHandler, param=[annotated_img, temp_img, pts])
    while len(pts) < 4:
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.waitKey(250)
    cv2.destroyWindow(title)
    drawLines(annotated_img, pts[0])
    return pts


if __name__ == '__main__':

    params_dict = getParamDict()
    param_ids = readDistGridParams()
    pause_seq = 0
    gt_col = (0, 255, 0)
    actors=params_dict['actors']
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d.jpg'

    actor_id = param_ids['actor_id']
    seq_id = param_ids['seq_id']

    actor = actors[actor_id]
    sequences = sequences[actor]
    seq_name = sequences[seq_id]

    print 'actor: ', actor
    print 'seq_name: ', seq_name

    src_fname = db_root_dir + '/' + actor + '/' + seq_name + '/' + img_name_fmt
    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        print 'The video file ', src_fname, ' could not be opened'
        sys.exit()

    gt_corners_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    corrected_corners_fname = db_root_dir + '/' + actor + '/' + seq_name + '_corr.txt'

    ground_truth = readTrackingData(gt_corners_fname)
    no_of_frames = ground_truth.shape[0]
    print 'no_of_frames: ', no_of_frames

    gt_corners_window_name = 'Ground Truth Corners'
    cv2.namedWindow(gt_corners_window_name)

     cv2.setMouseCallback(gt_corners_window_name, mouseHandler, param=[annotated_img, temp_img, pts])

    for i in xrange(no_of_frames):


        curr_corners = np.asarray([ground_truth[i, 0:2].tolist(),
                                   ground_truth[i, 2:4].tolist(),
                                   ground_truth[i, 4:6].tolist(),
                                   ground_truth[i, 6:8].tolist()]).T
        ret, curr_img = cap.read()
        if not ret:
            print 'End of sequence reached unexpectedly'
            break

        drawRegion(curr_img, curr_corners, gt_col, 1)
        cv2.imshow(gt_corners_window_name, curr_img)

        key = cv2.waitKey(1 - pause_seq)
        if key == 27:
            break
        elif key == 32:
            pause_seq = 1 - pause_seq














