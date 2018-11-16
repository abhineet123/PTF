import utility as util
import numpy as np
import os
import sys
import cv2
import math

def readTrackingData(filename):
    if not os.path.isfile(filename):
        print "Tracking data file not found:\n ", filename
        sys.exit()

    data_file = open(filename, 'r')
    data_file.readline()
    lines = data_file.readlines()
    no_of_lines = len(lines)
    data_array = np.empty([no_of_lines, 8])
    line_id = 0
    for line in lines:
        # print(line)
        words = line.split()
        if (len(words) != 9):
            msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
            raise SyntaxError(msg)
        words = words[1:]
        coordinates = []
        for word in words:
            coordinates.append(float(word))
        data_array[line_id, :] = coordinates
        # print words
        line_id += 1
    data_file.close()
    return data_array


def drawRegion(img, corners, color, thickness=1):
    # draw the bounding box specified by the given corners
    for i in xrange(4):
        p1 = (int(corners[0, i]), int(corners[1, i]))
        p2 = (int(corners[0, (i + 1) % 4]), int(corners[1, (i + 1) % 4]))
        cv2.line(img, p1, p2, color, thickness)


def writeCorners(file_id, corners):
    # write the given corners to the file
    corner_str = ''
    for i in xrange(4):
        corner_str = corner_str + '{:5.2f}\t{:5.2f}\t'.format(corners[0, i], corners[1, i])
    file_id.write(corner_str + '\n')

if __name__ == '__main__':

    sequences = ['nl_bookI_s3',
                 'nl_bookII_s3',
                 'nl_bookIII_s3',
                 'nl_bus',
                 'nl_cereal_s3',
                 'nl_highlighting',
                 'nl_juice_s3',
                 'nl_letter',
                 'nl_mugI_s3',
                 'nl_mugII_s3',
                 'nl_mugIII_s3',
                 'nl_newspaper']
    db_root_path = 'E:/UofA/Thesis/#Code/Datasets'
    actor = 'Human'
    seq_id = 4
    update_base_corners = 0


    seq_name = sequences[seq_id]
    src_fname = db_root_path + '/' + actor + '/' + seq_name + '/img%03d.jpg'
    ground_truth_fname = db_root_path + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)
    affine_fname = db_root_path + '/' + actor + '/' + seq_name + '_affine.txt'

    affine_file = open(affine_fname, 'w')
    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        print 'The video file ', src_fname, ' could not be opened'
        sys.exit()

    no_of_frames = ground_truth.shape[0]

    base_corners = np.asarray([ground_truth[0, 0:2].tolist(),
                               ground_truth[0, 2:4].tolist(),
                               ground_truth[0, 4:6].tolist(),
                               ground_truth[0, 6:8].tolist()]).T
    print 'base_corners: ', base_corners

    writeCorners(affine_file, base_corners)

    ret, init_img = cap.read()

    window_name = 'Ground Truths'
    cv2.namedWindow(window_name)

    act_col = (0, 0, 0)
    hom_col = (0, 0, 255)
    affine_col = (0, 255, 0)
    rt_col = (255, 0, 0)
    trans_col = (255, 255, 255)
    comp_col = (255, 255, 0)

    hom_errors = []
    affine_errors = []
    rt_errors = []
    trans_errors = []

    for i in xrange(1, no_of_frames):
        current_corners = np.asarray([ground_truth[i, 0:2].tolist(),
                                      ground_truth[i, 2:4].tolist(),
                                      ground_truth[i, 4:6].tolist(),
                                      ground_truth[i, 6:8].tolist()]).T
        H = util.compute_homography(base_corners, current_corners)

        H_inv = np.linalg.inv(np.mat(H))
        # H_inv=H_inv/H_inv[2, 2]

        v1=H_inv[2, 0]
        v2=H_inv[2, 1]

        u=H_inv[2, 2]

        dx = H_inv[0, 2]
        dy = H_inv[1, 2]
        dxx = H_inv[0, 0] - H_inv[2, 0] * H_inv[0, 2] - 1
        dxy = H_inv[0, 1] - H_inv[2, 1] * H_inv[0, 2]
        dyx = H_inv[1, 0] - H_inv[2, 0] * H_inv[1, 2]
        dyy = H_inv[1, 1] - H_inv[2, 1] * H_inv[1, 2] - 1


        affine_mat_inv = np.array(
            [[1 + dxx, dxy, dx],
             [dyx, 1 + dyy, dy],
             [0, 0, 1]]
        )
        P_inv=np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [v1, v2, u]]
        )

        H_inv_comp = np.mat(affine_mat_inv) * np.mat(P_inv)

        print 'H_inv:\n', H_inv
        print 'H_inv_comp:\n', H_inv_comp

        affine_mat = np.linalg.inv(np.mat(affine_mat_inv))
        P = np.linalg.inv(np.mat(P_inv))
        H_comp = np.mat(P) * np.mat(affine_mat)
        H_comp2 = np.mat(affine_mat) * np.mat(P)

        print 'H:\n', np.array(H)
        print 'H_comp:\n', H_comp
        print 'H_comp2:\n', H_comp2
        print 'affine_mat:\n', affine_mat
        print 'P:\n', P




        a = np.sqrt((dxx + 1) * (dxx + 1) + (dyx * dyx))-1
        cos_theta = (dxx + 1) / (a + 1)
        sin_theta = (dyx) / (a + 1)
        s = (dxy * cos_theta + (dyy + 1) * sin_theta) / ((dyy + 1) * cos_theta - dxy * sin_theta)
        b = (dxy / (s * cos_theta - sin_theta)) - 1

        # print 'a: ', a
        # print 'b: ', b
        # print 's: ', s
        # print 'cos_theta: ', cos_theta
        # print 'sin_theta: ', sin_theta


        rt_mat = np.array(
            [[cos_theta, - sin_theta, dx],
             [sin_theta, cos_theta, dy],
             [0, 0, 1]]
        )
        trans_mat = np.array(
            [[1, 0, dx],
             [0, 1, dy],
             [0, 0, 1]]
        )

        # print 'dx: ', dx
        # print 'dy: ', dy
        # print 'dxx: ', dxx
        # print 'dxy: ', dxy
        # print 'dyx: ', dyx
        # print 'dyy: ', dyy
        # print 'affine_mat:\n', affine_mat
        # print 'rt_mat:\n', rt_mat
        # print 'trans_mat:\n', trans_mat

        base_corners_hm = util.homogenize(base_corners)
        # print 'base_corners_hm: ', base_corners_hm

        hom_corners_hm = np.mat(H) * np.mat(base_corners_hm)
        hom_corners = util.dehomogenize(hom_corners_hm)
        hom_error = math.sqrt(np.sum(np.square(hom_corners - current_corners)) / 4)
        hom_errors.append(hom_error)

        affine_corners_hm = np.mat(affine_mat) * np.mat(base_corners_hm)
        affine_corners = util.dehomogenize(affine_corners_hm)
        affine_error = math.sqrt(np.sum(np.square(affine_corners - current_corners)) / 4)
        affine_errors.append(affine_error)

        comp_corners_hm = np.mat(P) * np.mat(affine_mat) * np.mat(base_corners_hm)
        comp_corners = util.dehomogenize(comp_corners_hm)

        rt_corners_hm = np.mat(rt_mat) * np.mat(base_corners_hm)
        rt_corners = util.dehomogenize(rt_corners_hm)
        rt_error = math.sqrt(np.sum(np.square(rt_corners - current_corners)) / 4)
        rt_errors.append(rt_error)

        trans_corners_hm = np.mat(trans_mat) * np.mat(base_corners_hm)
        trans_corners = util.dehomogenize(trans_corners_hm)
        trans_error = math.sqrt(np.sum(np.square(trans_corners - current_corners)) / 4)
        trans_errors.append(trans_error)

        # print 'affine_corners_hm: ', affine_corners_hm
        print 'base_corners:\n', base_corners
        # print 'current_corners: ', current_corners
        print 'hom_corners_hm:\n', hom_corners_hm
        print 'hom_corners:\n', hom_corners
        print 'comp_corners:\n', comp_corners
        print 'affine_corners:\n', affine_corners
        print 'rt_corners:\n', rt_corners
        print 'trans_corners:\n', trans_corners

        ret, src_img = cap.read()

        # draw_region(src_img, current_corners, act_col, 1)
        drawRegion(src_img, hom_corners, hom_col, 1)
        drawRegion(src_img, affine_corners, affine_col, 1)
        drawRegion(src_img, comp_corners, comp_col, 1)
        # drawRegion(src_img, rt_corners, rt_col, 1)
        # drawRegion(src_img, trans_corners, trans_col, 1)

        # cv2.putText(src_img, "Actual", (int(current_corners[0, 0]), int(current_corners[1, 0])),
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, act_col)
        cv2.putText(src_img, "Homography", (int(hom_corners[0, 1]), int(hom_corners[1, 1])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, hom_col)
        cv2.putText(src_img, "Affine", (int(affine_corners[0, 2]), int(affine_corners[1, 2])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, affine_col)
        cv2.putText(src_img, "Composite", (int(comp_corners[0, 2]), int(comp_corners[1, 2])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, comp_col)
        # cv2.putText(src_img, "RT", (int(rt_corners[0, 3]), int(rt_corners[1, 3])),
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, rt_col)
        # cv2.putText(src_img, "Trans", (int(trans_corners[0, 3]), int(trans_corners[1, 3])),
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, rt_col)

        cv2.putText(src_img, "{:5.2f} {:5.2f} {:5.2f} {:5.2f}".format(hom_error, affine_error, rt_error, trans_error), (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

        cv2.imshow(window_name, src_img)

        writeCorners(affine_file, affine_corners)

        print '*'*100

        if update_base_corners:
            base_corners = np.copy(current_corners)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('p'):
            cv2.waitKey(0)

    mean_hom_error = np.mean(hom_errors)
    print 'mean_hom_error: ', mean_hom_error

    mean_affine_error = np.mean(affine_errors)
    print 'mean_affine_error: ', mean_affine_error

    mean_rt_error = np.mean(rt_errors)
    print 'mean_rt_error: ', mean_rt_error

    mean_trans_error = np.mean(trans_errors)
    print 'mean_trans_error: ', mean_trans_error

    affine_file.close()















