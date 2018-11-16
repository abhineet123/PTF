from decomposition import *
import time

if __name__ == '__main__':

    params_dict = getParamDict()
    param_ids = readDistGridParams()
    pause_seq = 0
    gt_col = (0, 255, 0)
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'
    actor = 'VTD'
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d.jpg'
    sequences = sequences[actor]

    for seq_id in xrange(0, 1):
        # seq_id = param_ids['seq_id']
        seq_name = sequences[seq_id]

        print 'seq_id: ', seq_id
        print 'seq_name: ', seq_name

        src_fname = db_root_dir + '/' + actor + '/' + seq_name + '/' + img_name_fmt
        cap = cv2.VideoCapture()
        if not cap.open(src_fname):
            print 'The video file ', src_fname, ' could not be opened'
            sys.exit()

        gt_corners_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
        gt_warp_fname = db_root_dir + '/' + actor + '/groundtruth_warps/' + seq_name + '.warps'
        gt_warps = readWarpData(gt_warp_fname)
        no_of_frames = gt_warps.shape[0]
        print 'no_of_frames: ', no_of_frames

        init_warp_inv = np.mat(gt_warps[0, :].reshape((3, 3), order='C'))
        init_warp = np.linalg.inv(init_warp_inv)

        ret, init_img = cap.read()
        # init_corners = getTrackingObject2(init_img, col=(0, 0, 255), title='Select initial object location')
        # init_corners = np.asarray(init_corners).T
        # base_corners_hm = init_warp_inv * util.homogenize(init_corners)

        reference_corners = np.array([
            [76.199996948, 76.199996948],
            [279.399993896, 76.199996948],
            [279.399993896, 215.899993896],
            [76.199996948, 215.899993896]
        ]).T
        init_corners = util.dehomogenize(init_warp * util.homogenize(reference_corners))
        new_corners = arrangeCorners(init_corners)

        drawRegion(init_img, new_corners, gt_col, 1)


        # print 'init_warp:\n', init_warp
        # # print 'base_corners_hm:\n', base_corners_hm
        # print 'reference_corners:\n', reference_corners
        # print 'init_corners:\n', init_corners

        gt_corners_file = open(gt_corners_fname, 'w')
        gt_corners_file.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
        writeCorners(gt_corners_file, new_corners, 1)

        gt_corners_window_name = 'Ground Truth Corners'
        cv2.namedWindow(gt_corners_window_name)
        cv2.imshow(gt_corners_window_name, init_img)

        for i in xrange(1, no_of_frames):
            key = cv2.waitKey(1 - pause_seq)
            if key == 27:
                break
            elif key == 32:
                pause_seq = 1 - pause_seq

            curr_warp_inv = np.mat(gt_warps[i, :].reshape((3, 3), order='C'))
            curr_warp = np.linalg.inv(curr_warp_inv)

            # print 'curr_warp_inv:\n', curr_warp_inv
            # print 'curr_warp:\n', curr_warp
            # curr_corners = util.homogenize(curr_warp * base_corners_hm)
            curr_corners = util.dehomogenize(curr_warp * util.homogenize(reference_corners))
            new_corners = arrangeCorners(curr_corners)

            # print 'reference_corners:\n', reference_corners
            # print 'curr_corners:\n', curr_corners
            # print 'new_corners:\n', new_corners

            ret, curr_img = cap.read()
            if not ret:
                print 'End of sequence reached unexpectedly'
                break

            writeCorners(gt_corners_file, new_corners, i + 1)
            drawRegion(curr_img, curr_corners, gt_col, 1)
            cv2.imshow(gt_corners_window_name, curr_img)

        gt_corners_file.close()















