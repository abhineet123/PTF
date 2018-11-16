from distanceGrid import *
import time
import os
from Misc import *
import shutil
import Metaio

if __name__ == '__main__':

    actor = 'METAIO'

    params_dict = getParamDict()
    sequences = params_dict['sequences']
    challenges = params_dict['challenges']

    db_root_dir = '../Datasets'
    track_root_dir = '../Tracking Data'
    img_root_dir = '../Image Data'
    dist_root_dir = '../Distance Data'

    dist_prarams = readDistGridParams()
    seq_id = dist_prarams['seq_id']
    challenge_id = dist_prarams['challenge_id']

    frame_ids = [0, 250, 500, 750, 1000]

    arg_id = 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        challenge_id = int(sys.argv[arg_id])
        arg_id += 1

    sequences = sequences[actor]

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

    seq_name = seq_name + '_' + challenge
    gt_file = db_root_dir + '/' + actor + '/' + seq_name + '.txt'
    gt_fid = open(gt_file, 'w')
    gt_fid.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')

    init_file = db_root_dir + '/' + actor + '/init/' + seq_name + '.txt'
    init_corners = readMetaioInitData(init_file)
    no_of_frames = init_corners.shape[0]

    if no_of_frames != len(frame_ids):
        print 'unexpected frame count in init file: ', no_of_frames
        sys.exit()

    corner_id = 0
    template_corners_corrected = np.zeros([2, 4], dtype=np.float64)
    for frame_id in frame_ids:
        template_corners = init_corners[corner_id, :].reshape([4, 2]).transpose()
        # print 'template_corners:\n', template_corners

        template_corners_corrected[0, 0] = template_corners[0][1]
        template_corners_corrected[1, 0] = template_corners[1][1]

        template_corners_corrected[0, 1] = template_corners[0][0]
        template_corners_corrected[1, 1] = template_corners[1][0]

        template_corners_corrected[0, 2] = template_corners[0][3]
        template_corners_corrected[1, 2] = template_corners[1][3]

        template_corners_corrected[0, 3] = template_corners[0][2]
        template_corners_corrected[1, 3] = template_corners[1][2]

        template_corners_scaled = Metaio.toTemplate.region(template_corners_corrected)
        # print 'template_corners_scaled:\n', template_corners_scaled
        writeCorners(gt_fid, template_corners_scaled, frame_id + 1)
        corner_id += 1

    gt_fid.close()