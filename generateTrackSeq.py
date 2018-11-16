import numpy as np
import cv2
import math
import os


def rotatePoint(orig_pt, center, theta):
    orig_pt = np.mat(orig_pt).transpose()
    center = np.mat(center).transpose()

    # print 'orig_pt: ', orig_pt
    # print 'center: ', center
    rot_mat = np.mat(
        [[math.cos(theta), - math.sin(theta)],
         [math.sin(theta), math.cos(theta)]]
    )
    rot_pt = rot_mat * (orig_pt - center) + center

    return rot_pt[0, 0], rot_pt[1, 0]


def mouseCallback(event, x, y, flags, param):
    global rot_acw, rot_cw
    global trans_up, trans_down, trans_right, trans_left
    global exit_loop, line_changed
    global speed_x, speed_y

    if event == cv2.EVENT_LBUTTONDOWN:
        rot_acw = 1
    elif event == cv2.EVENT_LBUTTONUP:
        rot_acw = 0
    elif event == cv2.EVENT_RBUTTONDOWN:
        rot_cw = 1
    elif event == cv2.EVENT_RBUTTONUP:
        rot_cw = 0


if __name__ == '__main__':
    img_height = 600
    img_width = 800
    no_of_frames = 5000

    bkg_col = (255, 255, 255)
    frg_col = (0, 0, 0)

    line_length = 100
    line_thickness = 20

    speed_x = 2.0
    speed_y = 2.0
    dspeed_x = 0.5
    dspeed_y = 0.5

    speed_rot = 0.1
    dspeed_rot = 0.1

    center_x = 100
    center_y = 100
    start_x = center_x - line_length / 2
    start_y = center_y
    end_x = center_x + line_length / 2
    end_y = center_y

    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    img[:, :, 0].fill(bkg_col[0])
    img[:, :, 1].fill(bkg_col[1])
    img[:, :, 2].fill(bkg_col[2])

    output_dir = 'img_seq'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # vid_fname = 'img_seq.avi'
    # vid_writer = cv2.VideoWriter()
    # if vid_writer.open(vid_fname, cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 30, (600, 800)) != 0:
    # raise SyntaxError('Video file could not be opened')

    rot_cw = 0
    rot_acw = 0
    trans_right = 0
    trans_left = 0
    trans_up = 0
    trans_down = 0

    line_changed = 1
    exit_loop = 0

    frame_id = 0
    win_name = 'Line Sequence'
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouseCallback)

    while not exit_loop:

        cv2.line(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), frg_col, line_thickness,
                 lineType=cv2.cv.CV_AA)
        cv2.imshow(win_name, img)

        k = cv2.waitKey(1)
        print 'key: ', k
        if k == 27:
            exit_loop = 1
        elif k == ord('s') or k == 2621440:
            trans_down = 1
            trans_up = 0
            line_changed = 1
        elif k == ord('w') or k == 2490368:
            trans_up = 1
            trans_down = 0
            line_changed = 1
        elif k == ord('d') or k == 2555904:
            trans_right = 1
            trans_left = 0
            line_changed = 1
        elif k == ord('a') or k == 2424832:
            trans_left = 1
            trans_right = 0
            line_changed = 1
        elif k == ord('t'):
            rot_cw = 1
            line_changed = 1
        elif k == ord('r'):
            rot_acw = 1
            line_changed = 1
        elif k == ord(']'):
            speed_x += dspeed_x
            speed_y += dspeed_y
        elif k == ord('['):
            speed_x -= dspeed_x
            speed_y -= dspeed_y
            if speed_x < 0.1:
                speed_x = 0.1
            if speed_y < 0.1:
                speed_y = 0.1

        if rot_cw:
            start_x, start_y = rotatePoint([start_x, start_y], [center_x, center_y], speed_rot)
            end_x, end_y = rotatePoint([end_x, end_y], [center_x, center_y], speed_rot)
        if rot_acw:
            start_x, start_y = rotatePoint([start_x, start_y], [center_x, center_y], -speed_rot)
            end_x, end_y = rotatePoint([end_x, end_y], [center_x, center_y], -speed_rot)
        if trans_up:
            # trans_up = 0
            start_y -= speed_y
            end_y -= speed_y
            center_y -= speed_y
        if trans_down:
            # trans_down = 0
            start_y += speed_y
            end_y += speed_y
            center_y += speed_y
        if trans_right:
            # trans_right = 0
            start_x += speed_x
            end_x += speed_x
            center_x += speed_x
        if trans_left:
            # trans_left = 0
            start_x -= speed_x
            end_x -= speed_x
            center_x -= speed_x

        # print 'start_pt: ', [start_x, start_y]
        # print 'end_pt: ', [end_x, end_y]


        if line_changed:
            frame_id += 1
            img_fname = '{:s}/frame{:05d}.jpg'.format(output_dir, frame_id)
            cv2.imwrite(img_fname, img)


        # vid_writer.write(img)

        img[:, :, 0].fill(bkg_col[0])
        img[:, :, 1].fill(bkg_col[1])
        img[:, :, 2].fill(bkg_col[2])

        cv2.waitKey(1)

    print 'no_of_frames: ', frame_id









