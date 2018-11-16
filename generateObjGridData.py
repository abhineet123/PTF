from distanceGrid import *
import time
import os
from Misc import getParamDict
import shutil

if __name__ == '__main__':

    db_root_dir = '../Datasets'
    track_root_dir = '../Tracking Data'
    img_root_dir = '../Image Data'
    dist_root_dir = '../Distance Data'

    params_dict = getParamDict()
    param_ids = readDistGridParams()

    actors = params_dict['actors']
    sequences = params_dict['sequences']
    grid_types = params_dict['grid_types']
    inc_types = params_dict['inc_types']
    appearance_models = params_dict['appearance_models']
    tracker_types = params_dict['tracker_types']
    filter_types = params_dict['filter_types']
    challenges = params_dict['challenges']

    actor_id = param_ids['actor_id']
    seq_id = param_ids['seq_id']
    grid_id = param_ids['grid_id']
    appearance_id = param_ids['appearance_id']
    inc_id = param_ids['inc_id']
    tracker_id = param_ids['tracker_id']
    start_id = param_ids['start_id']
    filter_id = param_ids['filter_id']
    kernel_size = param_ids['kernel_size']
    n_bins = param_ids['n_bins']
    challenge_id = param_ids['challenge_id']

    actor = actors[actor_id]
    sequences = sequences[actor]

    start_id = 1
    std_resx = 10
    std_resy = 10
    pt_thickness = 2
    line_thickness = 2

    n_pts = std_resx * std_resy

    arg_id = 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1

    if seq_id >= len(sequences):
        print 'Invalid dataset_id: ', seq_id
        sys.exit()

    seq_name = sequences[seq_id]

    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name

    tracking_data_root_dir = '../Tracking Data/0bject Grid'
    tracking_data_dir = '{:s}/{:s}'.format(tracking_data_root_dir, seq_name)

    if not os.path.exists(tracking_data_root_dir):
        os.mkdir(tracking_data_root_dir)

    if not os.path.exists(tracking_data_dir):
        os.mkdir(tracking_data_dir)

    src_folder = db_root_dir + '/' + actor + '/' + seq_name

    ground_truth_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    ground_truth = readTrackingData(ground_truth_fname)
    no_of_frames = ground_truth.shape[0]
    print 'no_of_frames: ', no_of_frames
    end_id = no_of_frames

    std_pts, std_corners = getNormalizedUnitSquarePts(std_resx, std_resy, 0.5)
    std_pts_hm = util.homogenize(std_pts)

    horz_params = np.zeros([2, std_resy])
    vert_params = np.zeros([2, std_resx])
    horz_alpha = np.zeros([2, n_pts])
    vert_alpha = np.zeros([2, n_pts])

    prev_horz_alpha = np.zeros([2, n_pts])
    prev_vert_alpha = np.zeros([2, n_pts])

    horz_line_ids = np.zeros([2, std_resy])
    vert_line_ids = np.zeros([2, std_resx])

    for i in xrange(std_resx):
        vert_line_ids[0, i] = i * std_resy
        vert_line_ids[1, i] = (i + 1) * std_resy - 1

    for i in xrange(std_resy):
        horz_line_ids[0, i] = i
        horz_line_ids[1, i] = (std_resx - 1) * std_resy + i

    horz_line_ids = horz_line_ids.astype(np.uint32)
    vert_line_ids = vert_line_ids.astype(np.uint32)

    vert_params_file = '{:s}/vert_params.txt'.format(tracking_data_dir)
    horz_params_file = '{:s}/horz_params.txt'.format(tracking_data_dir)
    vert_alpha_file = '{:s}/vert_alpha.txt'.format(tracking_data_dir)
    horz_alpha_file = '{:s}/horz_alpha.txt'.format(tracking_data_dir)
    prev_vert_alpha_file = '{:s}/prev_vert_alpha.txt'.format(tracking_data_dir)
    prev_horz_alpha_file = '{:s}/prev_horz_alpha.txt'.format(tracking_data_dir)
    vert_alpha_diff_file = '{:s}/vert_alpha_diff.txt'.format(tracking_data_dir)
    horz_alpha_diff_file = '{:s}/horz_alpha_diff.txt'.format(tracking_data_dir)

    vert_params_fid = open(vert_params_file, 'w')
    horz_params_fid = open(horz_params_file, 'w')
    vert_alpha_fid = open(vert_alpha_file, 'w')
    horz_alpha_fid = open(horz_alpha_file, 'w')
    prev_vert_alpha_fid = open(prev_vert_alpha_file, 'w')
    prev_horz_alpha_fid = open(prev_horz_alpha_file, 'w')
    vert_alpha_diff_fid = open(vert_alpha_diff_file, 'w')
    horz_alpha_diff_fid = open(horz_alpha_diff_file, 'w')

    window_name = 'Piecewise Optimization'
    cv2.namedWindow(window_name)

    prev_pts = None

    for frame_id in xrange(start_id, end_id):
        # ret, curr_img = cap.read()
        curr_img = cv2.imread(src_folder + '/frame{:05d}.jpg'.format(frame_id + 1))
        # print 'curr_img: ', curr_img
        if curr_img is None:
            break

        start_time = time.clock()

        curr_corners = np.asarray([ground_truth[frame_id, 0:2].tolist(),
                                   ground_truth[frame_id, 2:4].tolist(),
                                   ground_truth[frame_id, 4:6].tolist(),
                                   ground_truth[frame_id, 6:8].tolist()]).T

        curr_corners_norm, curr_norm_mat = getNormalizedPoints(curr_corners)
        try:
            curr_hom_mat = np.mat(util.compute_homography(std_corners, curr_corners))
        except np.linalg.linalg.LinAlgError as error_msg:
            print'Error encountered while computing homography for frame {:d}: {:s}'.format(frame_id, error_msg)
            break
        # curr_pts_norm = util.dehomogenize(curr_hom_mat * std_pts_hm)
        # curr_pts = util.dehomogenize(curr_norm_mat * util.homogenize(curr_pts_norm))

        curr_pts = util.dehomogenize(curr_hom_mat * std_pts_hm)

        # print 'curr_pts_int: \n', curr_pts_int

        # drawRegion(curr_img, curr_corners, (0, 255, 0))


        for i in xrange(std_resx):
            start_pt = curr_pts[:, horz_line_ids[0, i]]
            end_pt = curr_pts[:, horz_line_ids[1, i]]
            m = (end_pt[1] - start_pt[1]) / (end_pt[0] - start_pt[0])
            c = start_pt[1] - m * start_pt[0]
            horz_params[0, i] = m
            horz_params[1, i] = c

            cv2.line(curr_img, (int(start_pt[0]), int(start_pt[1])), (int(end_pt[0]), int(end_pt[1])),
                     (0, 0, 255), thickness=line_thickness)

        for i in xrange(std_resy):
            start_pt = curr_pts[:, vert_line_ids[0, i]]
            end_pt = curr_pts[:, vert_line_ids[1, i]]
            m = (end_pt[1] - start_pt[1]) / (end_pt[0] - start_pt[0])
            c = start_pt[1] - m * start_pt[0]
            vert_params[0, i] = m
            vert_params[1, i] = c
            cv2.line(curr_img, (int(start_pt[0]), int(start_pt[1])), (int(end_pt[0]), int(end_pt[1])),
                     (0, 255, 0), thickness=line_thickness)

        vert_params.flatten(order='C').tofile(vert_params_fid, sep='\t', format='%12.6f')
        vert_params_fid.write('\n')
        horz_params.flatten(order='C').tofile(horz_params_fid, sep='\t', format='%12.6f')
        horz_params_fid.write('\n')

        for i in xrange(n_pts):
            curr_pt = curr_pts[:, i]

            vert_id = int(math.floor(i / std_resy))
            vert_start_id = vert_line_ids[0, vert_id]
            vert_end_id = vert_line_ids[1, vert_id]

            vert_start_pt = curr_pts[:, vert_line_ids[0, vert_id]]
            vert_end_pt = curr_pts[:, vert_line_ids[1, vert_id]]
            vert_alpha[:, i] = (curr_pt - vert_start_pt) / (vert_end_pt - vert_start_pt)

            horz_id = i % std_resy
            horz_start_id = horz_line_ids[0, horz_id]
            horz_end_id = horz_line_ids[1, horz_id]
            horz_start_pt = curr_pts[:, horz_start_id]
            horz_end_pt = curr_pts[:, horz_end_id]
            horz_alpha[:, i] = (curr_pt - horz_start_pt) / (horz_end_pt - horz_start_pt)

            if prev_pts is not None:
                prev_pt = prev_pts[:, i]
                vert_start_pt = prev_pts[:, vert_line_ids[0, vert_id]]
                vert_end_pt = prev_pts[:, vert_line_ids[1, vert_id]]
                prev_vert_alpha[:, i] = (prev_pt - vert_start_pt) / (vert_end_pt - vert_start_pt)

                horz_start_pt = prev_pts[:, horz_start_id]
                horz_end_pt = prev_pts[:, horz_end_id]
                prev_horz_alpha[:, i] = (prev_pt - horz_start_pt) / (horz_end_pt - horz_start_pt)
            else:
                prev_vert_alpha[:, i] = np.copy(vert_alpha[:, i])
                prev_horz_alpha[:, i] = np.copy(horz_alpha[:, i])

            cv2.circle(curr_img, (int(curr_pt[0]), int(curr_pt[1])), 2, (255, 0, 0), thickness=pt_thickness)

            # print 'i={:3d} vert_id={:3d} vert_start_id={:3d} vert_end_id={:3d} horz_id={:3d} horz_start_id={:3d} horz_end_id={:3d}'.format(
            # i, vert_id, vert_start_id, vert_end_id, horz_id, horz_start_id, horz_end_id)
            #
            # print 'vert_alpha: ', vert_alpha[:, i].flatten()
            # print 'horz_alpha: ', horz_alpha[:, i].flatten()
            #
            # print 'curr_pt: ', curr_pt.flatten()
            # print 'vert_start_pt: ', vert_start_pt.flatten()
            # print 'vert_end_pt: ', vert_end_pt.flatten()
            # print 'horz_start_pt: ', horz_start_pt.flatten()
            # print 'horz_end_pt: ', horz_end_pt.flatten()

        vert_alpha.flatten(order='F').tofile(vert_alpha_fid, sep='\t', format='%12.6f')
        vert_alpha_fid.write('\n')
        horz_alpha.flatten(order='F').tofile(horz_alpha_fid, sep='\t', format='%12.6f')
        horz_alpha_fid.write('\n')

        prev_vert_alpha.flatten(order='F').tofile(prev_vert_alpha_fid, sep='\t', format='%12.6f')
        prev_vert_alpha_fid.write('\n')
        prev_horz_alpha.flatten(order='F').tofile(prev_horz_alpha_fid, sep='\t', format='%12.6f')
        prev_horz_alpha_fid.write('\n')

        vert_alpha_diff = prev_vert_alpha - vert_alpha
        horz_alpha_diff = prev_horz_alpha - horz_alpha

        vert_alpha_diff.flatten(order='F').tofile(vert_alpha_diff_fid, sep='\t', format='%12.6f')
        vert_alpha_diff_fid.write('\n')
        horz_alpha_diff.flatten(order='F').tofile(horz_alpha_diff_fid, sep='\t', format='%12.6f')
        horz_alpha_diff_fid.write('\n')

        end_time = time.clock()
        curr_time = end_time - start_time
        curr_fps = 1.0 / curr_time

        prev_pts = curr_pts.copy()

        print 'frame_id:\t{:-5d}\tTime:\t{:-14.10f}'.format(frame_id, curr_time)

        cv2.imshow(window_name, curr_img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    vert_params_fid.close()
    horz_params_fid.close()
    vert_alpha_fid.close()
    horz_alpha_fid.close()
    prev_vert_alpha_fid.close()
    prev_horz_alpha_fid.close()
    vert_alpha_diff_fid.close()
    horz_alpha_diff_fid.close()

    print 'Exiting...'

