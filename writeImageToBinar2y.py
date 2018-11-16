from DecompUtils import *

if __name__ == '__main__':
    filter_types = {0: 'none',
                    1: 'gauss',
                    2: 'box',
                    3: 'norm_box',
                    4: 'bilateral',
                    5: 'median',
                    6: 'gabor',
                    7: 'sobel',
                    8: 'scharr',
                    9: 'LoG',
                    10: 'DoG',
                    11: 'laplacian',
                    12: 'canny'
    }

    sequences = {0: 'nl_bookI_s3',
                 1: 'nl_bookII_s3',
                 2: 'nl_bookIII_s3',
                 3: 'nl_cereal_s3',
                 4: 'nl_juice_s3',
                 5: 'nl_mugI_s3',
                 6: 'nl_mugII_s3',
                 7: 'nl_mugIII_s3',
                 8: 'nl_bookI_s4',
                 9: 'nl_bookII_s4',
                 10: 'nl_bookIII_s4',
                 11: 'nl_cereal_s4',
                 12: 'nl_juice_s4',
                 13: 'nl_mugI_s4',
                 14: 'nl_mugII_s4',
                 15: 'nl_mugIII_s4',
                 16: 'nl_bus',
                 17: 'nl_highlighting',
                 18: 'nl_letter',
                 19: 'nl_newspaper',
    }
    db_root_path = 'E:/UofA/Thesis/#Code/Datasets'
    actor = 'Human'
    seq_id = 0
    filter_id = 0
    kernel_size = 9
    start_id = 1
    show_img = 0

    arg_id = 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        filter_id = int(sys.argv[arg_id])
        arg_id += 1

    if seq_id >= len(sequences):
        print 'Invalid dataset_id: ', seq_id
        sys.exit()
    if filter_id >= len(filter_types):
        print 'Invalid filter_id: ', filter_id
        sys.exit()

    seq_name = sequences[seq_id]
    filter_type = filter_types[filter_id]
    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name
    print 'filter_type: ', filter_type
    print 'kernel_size: ', kernel_size
    src_folder = db_root_path + '/' + actor + '/' + seq_name
    img_folder = 'Image Data'
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    if filter_type != 'none':
        img_fname = img_folder + '/' + seq_name + '_' + filter_type + str(kernel_size) + '.bin'
    else:
        img_fname = img_folder + '/' + seq_name + '.bin'
    img_fid = open(img_fname, 'wb')

    ground_truth_fname = db_root_path + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)
    no_of_frames = ground_truth.shape[0]

    print 'no_of_frames: ', no_of_frames

    end_id = no_of_frames

    win_name = 'Filtered Image'
    if show_img:
        cv2.namedWindow(win_name)

    for frame_id in xrange(start_id, end_id):
        print 'frame_id: ', frame_id
        curr_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id + 1))
        curr_img_gs = cv2.cvtColor(curr_img, cv2.cv.CV_BGR2GRAY).astype(np.float64)
        if filter_type != 'none':
            curr_img_gs = applyFilter(curr_img_gs, filter_type, kernel_size)
        curr_img_gs.astype(np.uint8).tofile(img_fid)
        if show_img:
            cv2.imshow(win_name, curr_img_gs)
            if cv2.waitKey(1) == 27:
                break




