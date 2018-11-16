from distanceGrid import *
import time
import os
from Misc import getParamDict

if __name__ == '__main__':

    db_root_path = 'E:/UofA/Thesis/#Code/Datasets'
    actor = 'Human'

    params_dict = getParamDict()
    param_ids = readDistGridParams()

    sequences = params_dict['sequences']
    appearance_models = params_dict['appearance_models']
    filter_types = params_dict['filter_types']
    seq_id = 8
    appearance_model = 'mi_new'
    std_resx = 50
    std_resy = 50
    filter_id = 0
    kernel_size = 9
    start_id = 0

    n_pts = std_resx * std_resy
    n_bins = 8

    seq_name = sequences[seq_id]
    filter_type = filter_types[filter_id]

    dist_func, pre_proc_func, post_proc_func, opt_func = getDistanceFunction(appearance_model, n_pts, n_bins)

    src_folder = db_root_path + '/' + actor + '/' + seq_name

    ground_truth_fname = db_root_path + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)

    no_of_frames = ground_truth.shape[0]
    print 'no_of_frames: ', no_of_frames

    end_id = no_of_frames

    std_pts, std_corners = getNormalizedUnitSquarePts(std_resx, std_resy)
    std_pts_hm = util.homogenize(std_pts)
    std_corners_hm = util.homogenize(std_corners)

    for frame_id in xrange(start_id, end_id):

        print 'frame_id: ', frame_id

        curr_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id + 1))
        if curr_img is None:
            break
        curr_img_gs = cv2.cvtColor(curr_img, cv2.cv.CV_BGR2GRAY).astype(np.float64)
        if filter_type != 'none':
            curr_img_gs = applyFilter(curr_img_gs, filter_type, kernel_size)

        curr_img_gs = pre_proc_func(curr_img_gs)

        curr_corners = np.asarray([ground_truth[frame_id, 0:2].tolist(),
                                   ground_truth[frame_id, 2:4].tolist(),
                                   ground_truth[frame_id, 4:6].tolist(),
                                   ground_truth[frame_id, 6:8].tolist()]).T

        curr_corners_norm, curr_norm_mat = getNormalizedPoints(curr_corners)

        try:
            curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners_norm))
        except np.linalg.linalg.LinAlgError as error_msg:
            print'Error encountered while computing homography for frame {:d}: {:s}'.format(frame_id, error_msg)
            break
        curr_pts_norm = util.dehomogenize(curr_hom_mat * std_pts_hm)
        curr_pts = util.dehomogenize(curr_norm_mat * util.homogenize(curr_pts_norm))

        curr_pixel_vals = np.mat([util.bilin_interp(curr_img_gs, curr_pts[0, pt_id], curr_pts[1, pt_id]) for pt_id in
                                  xrange(n_pts)])
        curr_pixel_vals = post_proc_func(curr_pixel_vals)

        hist12, hist1, hist2=distanceUtils.getHistograms(curr_pixel_vals, curr_pixel_vals)
        curr_mi=dist_func(curr_pixel_vals, curr_pixel_vals)

        print 'hist12: \n', hist12
        print 'hist1: \n', hist1
        print 'hist2: \n', hist2
        print 'curr_mi: ', curr_mi

        k=raw_input('Press any key to continue')





