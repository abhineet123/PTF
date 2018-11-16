from Misc import getParamDict
from Misc import readTrackingDataMOT
from Misc import writeCornersMOT
from Misc import getFileList
import os
import sys
import shutil
import numpy as np

if __name__ == '__main__':

    params_dict = getParamDict()
    mot_actors = params_dict['mot_actors']
    mot_sequences = params_dict['mot_sequences']

    split_images = 0
    fix_frame_ids = 1

    n_split_seq = 13
    # n_split_seq = 15
    # n_split_seq = 30

    actor_id = 2
    seq_type_id = 0
    seq_id = 1

    # actor = None
    # seq_name = None
    actor = 'LOST'
    seq_name = '009_2011-03-29_07-00-00'
    # seq_name = 'M-30'
    # seq_name = 'M-30-HD'
    # seq_name = 'Urban1'


    # db_root_dir = '../Datasets'
    db_root_dir = 'C:/Datasets'

    # img_name_fmt = 'frame%05d.jpg'
    img_name_fmt = 'image%06d.jpg'
    img_ext = '.jpg'

    arg_id = 1
    if len(sys.argv) > arg_id:
        actor_id = int(sys.argv[arg_id])
        arg_id += 1
    if len(sys.argv) > arg_id:
        seq_id = int(sys.argv[arg_id])
        arg_id += 1

    if actor is None:
        actor = mot_actors[actor_id]
    if seq_name is None:
        sequences = mot_sequences[actor]
        if isinstance(sequences[0], list):
            sequences = sequences[seq_type_id]
        if seq_id >= len(sequences):
            print 'Invalid dataset_id: ', seq_id
            sys.exit()
        seq_name = sequences[seq_id]

    blobs_fname = db_root_dir + '/' + actor + '/' + seq_name + '/blobs.txt'
    if not os.path.isfile(blobs_fname):
        raise SyntaxError('File containing the tracks not found: {:s}'.format(tracks_fname))
    blobs_lines = open(blobs_fname, 'r').readlines()
    n_blobs_lines = len(blobs_lines)

    print 'actor: ', actor
    print 'seq_id:', seq_id, 'seq_name:', seq_name
    print 'n_blobs_lines: ', n_blobs_lines

    blobs_out_dir = db_root_dir + '/' + actor + '/Detections'
    if not os.path.exists(blobs_out_dir):
        os.mkdir(blobs_out_dir)
    blobs_out_fname = blobs_out_dir + '/' + seq_name + '.txt'
    blobs_out_file = open(blobs_out_fname, 'w')

    detections = np.zeros([n_blobs_lines, 5], np.float64)

    for line_id in xrange(n_blobs_lines):
        blob_line = blobs_lines[line_id]
        blob_words = blob_line
        blob_words = blob_line.split(' ')
        data = []
        if len(blob_words) != 5:
            msg = "Invalid formatting on line %d" % line_id + " in file %s" % blobs_fname + ":\n%s" % blob_line
            raise SyntaxError(msg)
        frame_id = int(blob_words[0]) + 1
        center_x = float(blob_words[1])
        center_y = float(blob_words[2])
        width = float(blob_words[3])
        height = float(blob_words[4])

        x = center_x - width/2.0
        y = center_y - height/2.0

        blobs_out_file.write('{:d}, -1, {:f}, {:f}, {:f}, {:f}, -1, -1, -1, -1\n'.format(
            frame_id, x, y, width, height))

        detections[line_id, 0] = frame_id
        detections[line_id, 1] = center_x
        detections[line_id, 2] = center_y
        detections[line_id, 3] = width
        detections[line_id, 4] = height
    blobs_out_file.close()

    tracks_fname = db_root_dir + '/' + actor + '/' + seq_name + '/tracks.txt'
    if not os.path.isfile(tracks_fname):
        raise SyntaxError('File containing the tracks not found: {:s}'.format(tracks_fname))
    tracks_lines = open(tracks_fname, 'r').readlines()
    n_tracks_lines = len(tracks_lines)

    print 'n_tracks_lines: ', n_tracks_lines

    tracks_out_dir = db_root_dir + '/' + actor + '/Annotations'
    if not os.path.exists(tracks_out_dir):
        os.mkdir(tracks_out_dir)
    tracks_out_fname = tracks_out_dir + '/' + seq_name + '.txt'
    tracks_out_file = open(tracks_out_fname, 'w')

    for line_id in xrange(n_tracks_lines):
        track_line = tracks_lines[line_id]
        track_words = track_line
        track_words = track_line.split(' ')
        data = []
        if len(track_words) != 4:
            msg = "Invalid formatting on line %d" % line_id + " in track file %s" % tracks_fname + ":\n%s" % track_line
            raise SyntaxError(msg)

        obj_id = int(track_words[0]) + 1
        frame_id = int(track_words[1]) + 1
        center_x = float(track_words[2])
        center_y = float(track_words[3])

        frame_detections = detections[detections[:, 0] == frame_id]
        n_detections = np.size(frame_detections, 0)
        min_dist = np.inf
        min_id = -1
        for detection_id in xrange(n_detections):
            detection_x = frame_detections[detection_id, 1]
            detection_y = frame_detections[detection_id, 2]

            dist_x = center_x - detection_x
            dist_y = center_y - detection_y

            dist = dist_x*dist_x + dist_y*dist_y
            if dist<min_dist:
                dist = min_dist
                min_id = detection_id
        width = frame_detections[detection_id, 3]
        height = frame_detections[detection_id, 4]

        x = center_x - width/2.0
        y = center_y - height/2.0

        tracks_out_file.write('{:d}, {:d}, {:f}, {:f}, {:f}, {:f}, -1, -1, -1, -1\n'.format(
            frame_id, obj_id, x, y, width, height))
    tracks_out_file.close()
































