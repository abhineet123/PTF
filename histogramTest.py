from distanceGrid import *
import utility as util
import numpy as np
import time
from Misc import getParamDict

if __name__ == '__main__':
    db_root_path = '../Datasets'
    actor = 'Human'
    params_dict = getParamDict()
    sequences = params_dict['sequences']

    hist_types = {0: 'floor',
                  1: 'round',
                  2: 'frac',
                  3: 'mi',
                  4: 'bspline',
    }

    hist_id = 4
    seq_id = 10

    frame_id1 = 1
    frame_id2 = 200

    std_resx = 100
    std_resy = 100
    n_pts = std_resx * std_resy
    min_bins = 2
    max_bins = 256
    base_bins = 8

    arg_id = 1
    if len(sys.argv) > arg_id:
        hist_id = int(sys.argv[arg_id])
        arg_id += 1

    hist_type = hist_types[hist_id]
    seq_name = sequences[seq_id]

    print 'seq_name: ',  seq_name
    print 'hist_type: ',  hist_type

    n_bins_vec = np.array(range(min_bins, max_bins + 1), dtype=np.uint32)
    n_bins_size = n_bins_vec.size

    src_folder = db_root_path + '/' + actor + '/' + seq_name

    ground_truth_fname = db_root_path + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)

    no_of_frames = ground_truth.shape[0]
    print 'no_of_frames: ', no_of_frames
    frame_id2_vec=range(2, no_of_frames)

    std_pts, std_corners = getNormalizedUnitSquarePts(std_resx, std_resy)
    std_pts_hm = util.homogenize(std_pts)
    std_corners_hm = util.homogenize(std_corners)

    corners = np.asarray([ground_truth[frame_id1 - 1, 0:2].tolist(),
                          ground_truth[frame_id1 - 1, 2:4].tolist(),
                          ground_truth[frame_id1 - 1, 4:6].tolist(),
                          ground_truth[frame_id1 - 1, 6:8].tolist()]).T
    (corners_norm, norm_mat) = getNormalizedPoints(corners)
    hom_mat = np.mat(util.compute_homography(std_corners, corners_norm))
    pts_norm = util.dehomogenize(hom_mat * std_pts_hm)
    pts = util.dehomogenize(norm_mat * util.homogenize(pts_norm))

    img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id1))
    img_gs = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY).astype(np.float64)

    corners2 = np.asarray([ground_truth[frame_id2 - 1, 0:2].tolist(),
                           ground_truth[frame_id2 - 1, 2:4].tolist(),
                           ground_truth[frame_id2 - 1, 4:6].tolist(),
                           ground_truth[frame_id2 - 1, 6:8].tolist()]).T
    (corners2_norm, norm_mat) = getNormalizedPoints(corners2)
    hom_mat = np.mat(util.compute_homography(std_corners, corners2_norm))
    pts2_norm = util.dehomogenize(hom_mat * std_pts_hm)
    pts2 = util.dehomogenize(norm_mat * util.homogenize(pts2_norm))

    img2 = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id2))
    img2_gs = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY).astype(np.float64)



    hist_fid = open('hist_{:s}.bin'.format(hist_type), 'wb')
    np.array([n_bins_size], dtype=np.uint32).tofile(hist_fid)
    n_bins_vec.tofile(hist_fid)

    for n_bins in n_bins_vec:

        distanceUtils.initStateVars(n_pts, n_bins)
        bin_centers = np.linspace(0, n_bins - 1, n_bins) * (base_bins - 1) / (n_bins - 1)

        pre_proc_func = lambda img: img * (n_bins - 1) / 255.0
        img_gs_norm = pre_proc_func(img_gs)
        pixel_vals = np.mat([util.bilin_interp(img_gs_norm, pts[0, pt_id], pts[1, pt_id]) for pt_id in
                             xrange(n_pts)])
        img2_gs_norm = pre_proc_func(img2_gs)

        pixel_vals2 = np.mat([util.bilin_interp(img2_gs_norm, pts2[0, pt_id], pts2[1, pt_id]) for pt_id in
                              xrange(n_pts)])
        start_time = time.clock()

        # mi_mat=distanceUtils.getMIMat(pixel_vals, pixel_vals2)
        # bin_centers.astype(np.float64).tofile(mi_mat_fid)
        # mi_mat.astype(np.float64).tofile(mi_mat_fid)

        if hist_type == 'floor':
            hist12, hist1, hist2 = distanceUtils.getHistogramsFloor(pixel_vals, pixel_vals2)
        elif hist_type == 'round':
            hist12, hist1, hist2 = distanceUtils.getHistogramsRound(pixel_vals, pixel_vals2)
        elif hist_type == 'frac':
            hist12, hist1, hist2 = distanceUtils.getfHistograms(pixel_vals, pixel_vals2)
        elif hist_type == 'bspline':
            # print 'calling getBSplineHistograms...'
            hist12, hist1, hist2 = distanceUtils.getBSplineHistograms(pixel_vals, pixel_vals2)
        elif hist_type == 'mi':
            hist12= distanceUtils.getMIMat(pixel_vals, pixel_vals2)

        end_time = time.clock()
        bin_centers.astype(np.float64).tofile(hist_fid)
        hist12.astype(np.float64).tofile(hist_fid)
        curr_time = end_time - start_time
        print 'n_bins:\t {:4d}\t hist time:\t {:15.12f}'.format(n_bins, curr_time)

        distanceUtils.freeStateVars()

    hist_fid.close()

    # np.savetxt('pixel_vals.txt', pixel_vals.transpose(), fmt='%12.8f', delimiter='\t')
    # np.savetxt('pixel_vals2.txt', pixel_vals2.transpose(), fmt='%12.8f', delimiter='\t')
    # np.savetxt('hist12.txt', hist12, fmt='%12.8f', delimiter='\t')
    # np.savetxt('fhist12.txt', fhist12, fmt='%12.8f', delimiter='\t')

    # print 'hist12:\n', hist12
    # print 'hist1:\n', hist1
    # print 'hist2:\n', hist2
    #
    # print 'fhist12:\n', fhist12
    # print 'fhist1:\n', fhist1
    # print 'fhist2:\n', fhist2