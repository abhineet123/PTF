from distanceGrid import *
import time
import os
from Misc import *
import shutil
import Metaio


def updateCanny(val):
    global canny_low_thresh, canny_ratio, update_canny
    canny_low_thresh = cv2.getTrackbarPos('canny_low_thresh', edge_win_name)
    canny_ratio = cv2.getTrackbarPos('canny_ratio', edge_win_name)
    update_canny = 1


def updateHough(val=-1):
    print 'in updateHough with val=', val
    global hough_thresh, hough_rho, hough_theta, update_hough
    hough_thresh = cv2.getTrackbarPos('thresh', line_win_name) + 1
    hough_rho = cv2.getTrackbarPos('rho', line_win_name) + 1
    hough_theta = float(cv2.getTrackbarPos('theta', line_win_name) + 1) / 1000.0
    update_hough = 1


def updateHoughTrackbar():
    global update_hough
    cv2.setTrackbarPos('thresh', line_win_name, hough_thresh - 1)
    cv2.setTrackbarPos('rho', line_win_name, hough_rho - 1)
    cv2.setTrackbarPos('theta', line_win_name, int(hough_theta * 1000 - 1))
    update_hough = 0


def updateHarris(val):
    global harris_blockSize, harris_ksize, harris_k, update_harris
    harris_blockSize = cv2.getTrackbarPos('blockSize', harris_win_name) + 1
    harris_ksize = 2 * cv2.getTrackbarPos('ksize', harris_win_name) + 1
    harris_k = float(cv2.getTrackbarPos('k', harris_win_name) + 1) / 1000.0
    update_harris = 1


def detectLines():
    # np.savetxt('curr_img_gs.txt', curr_img_gs, fmt='%12.6f', delimiter='\t')
    # curr_img_gs[curr_img_gs >= bkg_thresh] = 1
    no_of_lines = 0
    hough_thresh = 100
    lines = []
    while no_of_lines < 5:
        edge_img_copy = np.copy(edge_img)
        lines = cv2.HoughLines(edge_img_copy, hough_rho,
                               hough_theta, hough_thresh)
        if lines is None:
            break

        hough_thresh -= hough_thresh_diff
        lines = lines[0]
        no_of_lines = len(lines)

    # print 'lines:\n', lines[0]
    print 'no_of_lines:\n', no_of_lines
    print 'hough_thresh:\n', hough_thresh

    return lines, edge_img


def removeDuplicateLines(lines):
    out_lines = []
    min_diff_array = []
    # no_of_lines = len(lines)
    # print 'lines: ', lines
    for line in lines:
        r_in = line[0]
        theta_in = line[1]
        theta_in2 = cv2.cv.CV_PI - theta_in
        is_duplicate = 0
        min_diff = np.inf
        # print 'i: ', i
        # print 'line: ', line
        # if r_in < 0:
        # continue
        for out_line in out_lines:
            # print '\t j: ', j
            # print '\t out_line: ', out_line
            r_out = out_line[0]
            r_diff = math.fabs(r_out - r_in)
            r_diff2 = math.fabs(r_out + r_in)
            theta_diff = math.fabs(out_line[1] - theta_in)
            theta_diff2 = math.fabs(out_line[1] - theta_in2)
            if min_diff > r_diff:
                min_diff = r_diff
            if r_diff <= hough_r_min_diff and theta_diff <= hough_theta_min_diff:
                is_duplicate = 1
                break
            if r_diff2 <= hough_r_min_diff and theta_diff2 <= hough_theta_min_diff:
                is_duplicate = 1
                break
        # print 'min_diff: ', min_diff
        if not is_duplicate:
            # if np.isinf(min_diff):
            # min_diff=0
            out_lines.append([line[0], line[1]])
            min_diff_array.append(min_diff)
    no_of_lines = len(out_lines)
    # print 'before: no_of_lines: ', no_of_lines
    # print 'before: out_lines: ', out_lines
    # print 'min_diff_array: ', min_diff_array

    # min_diff_array = -1 * np.array(min_diff_array)
    # out_lines = np.array(out_lines)
    # out_lines = out_lines[min_diff_array.argsort()]

    # print 'after: out_lines: ', out_lines
    return out_lines


def getSobelEdges():
    sobel_x = cv2.Sobel(curr_img_gs, -1, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(curr_img_gs, -1, 0, 1, ksize=3)
    edge_img = np.zeros([img_height, img_width])
    for x in xrange(img_width):
        for y in xrange(img_height):
            if sobel_x[y, x] > 0 or sobel_y[y, x] > 0:
                edge_img[y, x] = 255
    return edge_img


if __name__ == '__main__':

    actor = 'METAIO'

    params_dict = getParamDict()
    sequences = params_dict['sequences']
    inc_types = params_dict['inc_types']
    appearance_models = params_dict['appearance_models']
    challenges = params_dict['challenges']
    start_id = 1

    db_root_dir = '../Datasets'
    track_root_dir = '../Tracking Data'
    img_root_dir = '../Image Data'
    dist_root_dir = '../Distance Data'

    dist_prarams = readDistGridParams()
    seq_id = dist_prarams['seq_id']
    inc_id = dist_prarams['inc_id']
    challenge_id = dist_prarams['challenge_id']

    arg_id = 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        challenge_id = int(sys.argv[arg_id])
        arg_id += 1

    if seq_id >= len(sequences):
        print 'Invalid seq_id: ', seq_id
        sys.exit()

    if challenge_id >= len(challenges):
        print 'Invalid challenge_id: ', challenge_id
        sys.exit()

    seq_name = sequences[seq_id]
    challenge = challenges[challenge_id]

    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    print 'challenge: ', challenge

    img_height = 640
    img_width = 480

    seq_name = seq_name + '_' + challenge
    src_folder = db_root_dir + '/' + actor + '/' + seq_name
    gt_file = db_root_dir + '/' + actor + '/' + seq_name + '._lines.txt'
    gt_fid = open(gt_file, 'w')

    init_file = db_root_dir + '/' + actor + '/init/' + seq_name + '.txt'

    init_corners = readMetaioInitData(init_file)
    template_corners = init_corners[0, :].reshape([4, 2]).transpose()
    template_corners_scaled = Metaio.toTemplate.region(template_corners)

    print 'init_corners: \n', init_corners
    print 'template_corners: \n', template_corners
    print 'template_corners_scaled: \n', template_corners_scaled
    # template_corners[1, :] = img_height - template_corners[1, :]
    # print 'template_corners: \n', template_corners

    gt_corners_file = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    gt_corners_fid = open(gt_corners_file, 'w')
    gt_corners_fid.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')

    file_list = os.listdir(src_folder)
    no_of_frames = len(file_list)
    end_id = no_of_frames

    print 'no_of_frames: ', no_of_frames

    edge_win_name = 'Edge Image'
    win_name = 'Binary Image'
    line_win_name = 'Line Image'
    corner_win_name = 'Corner Image'
    harris_win_name = 'Harris Image'

    cv2.namedWindow(win_name)
    cv2.namedWindow(line_win_name)
    cv2.namedWindow(edge_win_name)
    cv2.namedWindow(corner_win_name)

    bkg_thresh = 245

    hough_rho = 1
    hough_theta = cv2.cv.CV_PI / 180.0
    hough_theta_diff = 0.01
    hough_thresh = 100
    hough_thresh_diff = 1
    hough_srn = 0
    hough_stn = 0
    hough_min_length = 10

    hough_theta_min_diff = 0.20
    hough_r_min_diff = 100

    harris_blockSize = 32
    harris_ksize = 3
    harris_k = 0.1

    canny_low_thresh = 100
    canny_ratio = 10
    canny_aperture_size = 3

    line_cols = [(0, 0, 0),
                 (255, 0, 0),
                 (0, 255, 0),
                 (0, 0, 255)]

    cv2.createTrackbar('canny_low_thresh', edge_win_name, canny_low_thresh, 500, updateCanny)
    cv2.createTrackbar('canny_ratio', edge_win_name, canny_ratio, 100, updateCanny)

    cv2.createTrackbar('thresh', line_win_name, hough_thresh, 500, updateHough)
    cv2.createTrackbar('rho', line_win_name, hough_rho, 100, updateHough)
    cv2.createTrackbar('theta', line_win_name, int(hough_theta * 1000), 1000, updateHough)

    img_height = 0
    img_width = 0
    update_canny = 0
    update_hough = 0

    frame_id = 0
    last_frame_id = 1

    hough_theta_list = []
    hough_thresh_list = []

    pause_exec = 0
    use_harris = 0
    update_harris = 1
    use_prob_hough = 0

    if use_harris:
        cv2.namedWindow(harris_win_name)
        cv2.createTrackbar('blockSize', harris_win_name, harris_blockSize, 500, updateHarris)
        cv2.createTrackbar('ksize', harris_win_name, int((harris_ksize - 1) / 2), 10, updateHarris)
        cv2.createTrackbar('k', harris_win_name, int(harris_k * 1000), 1000, updateHarris)

    time_id = 0
    while frame_id < end_id:
        if last_frame_id != frame_id:
            print 'Updating frame {:d} time'.format(time_id)
            print 'frame_id:', frame_id
            print 'last_frame_id:', last_frame_id
            last_frame_id = frame_id
            update_canny = 1
            # update_hough = 1
            curr_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id + 1))
            if curr_img is None:
                print 'Frame: ', frame_id + 1, ' does not exist'
                break
            curr_img_gs = cv2.cvtColor(curr_img, cv2.cv.CV_BGR2GRAY).astype(np.uint8)
            img_height, img_width = curr_img_gs.shape
            curr_img_gs[curr_img_gs < bkg_thresh] = 0
            cv2.imshow(win_name, curr_img_gs)

        if update_canny:
            print 'Updating canny {:d} time'.format(time_id)
            update_hough = 1
            update_harris = 0
            update_canny = 0
            edge_img = cv2.Canny(curr_img_gs, canny_low_thresh, canny_low_thresh * canny_ratio,
                                 apertureSize=canny_aperture_size, L2gradient=True)
            # edge_img = getSobelEdges()
            # edge_img=cv2.Sobel(curr_img_gs, -1, 1, 1, ksize=3)
            # edge_img = cv2.cornerHarris(curr_img_gs, harris_blockSize, harris_ksize, harris_k)

        if use_harris and update_harris:
            update_harris = 0
            # print 'harris_blockSize: ', harris_blockSize
            # print 'harris_ksize: ', harris_ksize
            # print 'harris_k: ', harris_k
            harris_corners = cv2.cornerHarris(edge_img, harris_blockSize, harris_ksize, harris_k)
            harris_corners *= 255.0
            cv2.imshow(harris_win_name, harris_corners)
        elif update_hough and not use_harris:
            print 'entering Hough part {:d} time'.format(time_id)
            print 'update_hough: ', update_hough
            update_hough = 0
            edge_img_copy = np.copy(edge_img.astype(np.uint8))
            if hough_rho < 1:
                hough_rho = 1
            if hough_thresh < 1:
                hough_thresh = 1

            if use_prob_hough:
                lines = cv2.HoughLinesP(edge_img_copy, hough_rho,
                                        hough_theta, hough_thresh, minLineLength=hough_min_length)

                if lines is not None:
                    lines = lines[0]
                    print 'lines: ', lines
                    line_id = 0
                    curr_img_copy = np.copy(curr_img)
                    for line_params in lines:
                        print 'line_params: ', line_params

                        x1, y1, x2, y2 = line_params
                        txt_x = int((x1 + x2) / 2)
                        txt_y = int((y1 + y2) / 2)
                        # txt_y = 15 * (line_id + 2)
                        # txt_x = 10
                        cv2.line(curr_img_copy, (x1, y1), (x2, y2), thickness=2, color=line_cols[line_id % 4])
                        cv2.putText(curr_img_copy, 'line: {:d}'.format(line_id), (txt_x, txt_y),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, line_cols[line_id % 4])
                        cv2.putText(curr_img_copy, 'frame_id: {:d}'.format(frame_id), (10, 10),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0))
                        line_id += 1
            else:
                if pause_exec:
                    updateHough(0)
                    lines = cv2.HoughLines(edge_img_copy, hough_rho,
                                           hough_theta, hough_thresh)
                    if lines is not None:
                        lines = lines[0]
                        out_lines = removeDuplicateLines(lines)
                        no_of_lines = len(out_lines)
                        # print 'out_lines: \n', out_lines

                else:
                    print 'running cv2.HoughLines {:d} time'.format(time_id)
                    print 'update_hough: ', update_hough
                    time_id += 1
                    no_of_lines = 0
                    out_lines = None
                    hough_thresh = 100
                    while hough_thresh > 0:
                        no_of_lines = 0
                        hough_theta = cv2.cv.CV_PI / 180.0
                        while hough_theta > 0:
                            lines = cv2.HoughLines(edge_img_copy, hough_rho,
                                                   hough_theta, hough_thresh)
                            if lines is not None:
                                lines = lines[0]
                                out_lines = removeDuplicateLines(lines)
                                no_of_lines = len(out_lines)
                                if no_of_lines >= 4:
                                    break
                            hough_theta -= hough_theta_diff
                        if no_of_lines >= 4:
                            break
                        hough_thresh -= hough_thresh_diff

                    hough_theta_list.append(hough_theta)
                    hough_thresh_list.append(hough_thresh)

                    if no_of_lines < 4 or out_lines is None:
                        print 'Could not find enough lines in frame: ', frame_id + 1
                        # print 'out_lines: \n', out_lines
                        pause_exec = 1
                    elif no_of_lines > 4:
                        print 'Too many lines {:d} found in frame: {:d}'.format(no_of_lines, frame_id + 1)
                        # print 'out_lines: \n', out_lines
                        out_lines = out_lines[0:4]

                    updateHoughTrackbar()
                    # print 'hough_thresh: ', hough_thresh
                    # print 'hough_theta: ', hough_theta
                    # print 'hough_rho: ', hough_rho

                if not pause_exec:
                    out_lines = np.array(out_lines, dtype=np.float64)
                    # print 'final: out_lines: \n', out_lines
                    out_lines.flatten().tofile(gt_fid, sep='\t', format='%15.9f')
                    gt_fid.write('\n')
                    ulx, uly, urx, ury, lrx, lry, llx, lly = getIntersectionPoints(out_lines)
                    gt_corners_fid.write(
                        'frame{:5d}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}\n'.
                        format(frame_id + 1, ulx, uly, urx, ury, lrx, lry, llx, lly))
                    curr_img_copy2 = np.copy(curr_img)
                    corners = np.array([[ulx, urx, lrx, llx],
                                        [uly, ury, lry, lly]])
                    corners = refineCorners(corners, curr_img_gs)
                    # print 'corners: \n', corners
                    drawRegion(curr_img_copy2, corners, (0, 255, 0), 2)
                    drawRegion(curr_img_copy2, template_corners, (0, 0, 0), 2)
                    drawRegion(curr_img_copy2, template_corners_scaled, (0, 0, 0), 2)
                    cv2.imshow(corner_win_name, curr_img_copy2)
                # print 'lines: \n', lines

                # print 'no_of_lines: \n', no_of_lines
                # no_of_lines = len(lines)
                # print 'no_of_lines: ', no_of_lines
                # out_lines = removeDuplicateLines(lines)
                # print 'out_lines: \n', out_lines
                # print 'drawing lines: ', out_lines
                line_id = 0
                curr_img_copy = np.copy(curr_img)
                for line_params in out_lines:
                    # print 'line_params: ', line_params
                    r = line_params[0]
                    theta = line_params[1]
                    if theta == 0:
                        x1 = int(r)
                        x2 = x1
                        y1 = 0
                        y2 = img_height
                        m = c = np.inf
                    else:
                        m = -math.cos(theta) / math.sin(theta)
                        c = r / math.sin(theta)
                        x1 = 0
                        y1 = int(m * x1 + c)
                        x2 = img_width
                        y2 = int(m * x2 + c)
                    # txt_x = int((x1 + x2) / 2)
                    # txt_y = int((y1 + y2) / 2)
                    txt_y = 15 * (line_id + 2)
                    txt_x = 10
                    cv2.line(curr_img_copy, (x1, y1), (x2, y2), thickness=2, color=line_cols[line_id % 4])
                    cv2.putText(curr_img_copy,
                                '{:d} ({:6.2f},{:6.2f})({:6.2f},{:6.2f})({:6.2f},{:6.2f})({:6.2f},{:6.2f})'.
                                format(line_id, r, theta, m, c, x1, y1, x2, y2),
                                (txt_x, txt_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, line_cols[line_id % 4])
                    cv2.putText(curr_img_copy, 'frame_id: {:d}'.format(frame_id), (10, 10),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0))
                    line_id += 1
            cv2.imshow(line_win_name, curr_img_copy)
        cv2.imshow(edge_win_name, edge_img)

        # if not pause_exec:
        # frame_id += 1
        # frame_id -= 1

        key = cv2.waitKey(1)
        if key == ord('n'):
            frame_id += 1
        if key == ord('p'):
            frame_id -= 1
        elif key == 27:
            break
        elif key == 32:
            pause_exec = 1 - pause_exec

    gt_fid.close()
    gt_corners_fid.close()
