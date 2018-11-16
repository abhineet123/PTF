import numpy as np
import glob
import xml.etree.ElementTree as ET
import os
import sys
import json
import argparse


get_fn_without_ext = lambda fn: os.path.splitext(fn)[0]


def parse_pascal_voc(dir_name):
    # FOR ground truth bounding boxes only
    # Each frame K will have its ground truth detection in the file K.xml
    # For pascal voc format see 1.xml
    frames = dict()
    cur_dir = os.getcwd()
    os.chdir(dir_name)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.xml')

    for file in annotations:
        root = ET.parse(file).getroot()
        frame_id = get_fn_without_ext(file)
        R = dict(
            frame_id=frame_id,
            bboxes=[],
        )
        # actual parsing

        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            xmin = int(float(xmlbox.find('xmin').text))
            xmax = int(float(xmlbox.find('xmax').text))
            ymin = int(float(xmlbox.find('ymin').text))
            ymax = int(float(xmlbox.find('ymax').text))
            R['bboxes'].append([xmin, ymin, xmax, ymax])
        R['bboxes'] = np.asarray(R['bboxes'])
        R['detected'] = [False] * len(R['bboxes'])
        frames[str(frame_id)] = R

    os.chdir(cur_dir)
    return frames


def parse_json(dir_name):
    # For detection only
    # Frame K will have its detection in the file K.json in the folder.
    # Format: Check 1.json
    frames = list()
    cur_dir = os.getcwd()
    os.chdir(dir_name)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.json')

    for file in annotations:
        frame_id = get_fn_without_ext(file)
        bboxes = json.load(open(file))
        for bbox in bboxes:
            frames.append([frame_id, -1, bbox['topleft']['x'], bbox['topleft']['y'], bbox['bottomright']['x'],
                           bbox['bottomright']['y'], bbox['confidence']])

    os.chdir(cur_dir)
    return frames


def parse_txt_detection(fn):
    # assume MOT format:
    # frame_id, id, x, y, width, height, confidence
    data = np.genfromtxt(fn, dtype=str, delimiter=',')
    data = data[:, 0:7].astype(float)
    data[:, 4] = data[:, 2] + data[:, 4] - 1
    data[:, 5] = data[:, 3] + data[:, 5] - 1
    return data


def parse_txt_groundtruth(fn):
    # assume MOT format:
    # frame_id, id, x, y, width, height
    frames = dict()
    data = np.genfromtxt(fn, dtype=str, delimiter=',')
    data = data[:, 0:6].astype(float)
    for entry in data:
        frame_id, _, xmin, ymin, width, height = entry
        frame_id = int(frame_id)
        if str(frame_id) in frames:
            frames[str(frame_id)]['bboxes'].append([xmin, ymin, xmin + width - 1, ymin + height - 1])
            frames[str(frame_id)]['detected'].append(False)
        else:
            frames[str(frame_id)] = dict(
                frame_id=frame_id,
                bboxes=[[xmin, ymin, xmin + width - 1, ymin + height - 1]],
                detected=[False],
            )
    for frame in frames.values():
        frame['bboxes'] = np.asarray(frame['bboxes'])
    return frames


def compute_pre_rec(gt, detection, ovthresh=0.5):
    # detection format:
    # frame_id, id, xmin, ymin, xmax, ymax, confidence
    # ground truth format:
    # dictionary len for each frame id, each elem has bboxes K dictionary
    # 'bboxes' Kx4 ndarray: [xmin, ymin, xmax, ymax]
    #  'detected' K bool
    image_ids = [x[0] for x in detection]
    confidence = np.array([float(x[6]) for x in detection])
    BB = np.array([[float(z) for z in x[2:6]] for x in detection])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [int(image_ids[x]) for x in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    npos = 0
    for R in gt.values():
        npos += len(R['bboxes'])

    for d in range(nd):
        if str(image_ids[d]) not in gt:
            fp[d] = 1
            continue
        R = gt[str(image_ids[d])]
        BBGT = R['bboxes']
        bb = BB[d, :].astype(float)
        ovmax = -np.inf

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['detected'][jmax]:
                tp[d] = 1.
                R['detected'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.


    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return rec, prec, sorted_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_type', choices=['pascal_voc', 'txt'], type=str)
    parser.add_argument('gt_path', type=str)
    parser.add_argument('detection_type', choices=['json', 'txt'], type=str)
    parser.add_argument('detection_path', type=str)
    parser.add_argument('--ovthresh', default=0.5, type=float)
    parser.add_argument('--output_path', default='rec_prec_scores.pkl', type=str)
    args = parser.parse_args()
    return args


# This cript assume the frame id in both detection and ground truth are the same
# In pascal_voc and json reading the frame id will be deduced from the file names.
if __name__ == '__main__':
    if len(sys.argv) >= 5:
        args = parse_args()
        gt_type = args.gt_type
        gt_path = args.gt_path
        detection_type = args.detection_type
        detection_path = args.detection_path
        ovthresh = args.ovthresh
    else:
        seq_name = 'M-30-Large.txt'
        # seq_name = 'M-30-HD-Small.txt'
        # seq_name = 'M-30.txt'
        # seq_name = 'M-30-HD.txt'

        gt_type = 'txt'
        detection_type = 'txt'
        gt_path = 'log/gt/{:s}'.format(seq_name)
        detection_path = 'log/detections/{:s}'.format(seq_name)
        ovthresh = 0.5


    print 'gt_path: {:s}'.format(gt_path)
    if gt_type == 'pascal_voc':
        gt = parse_pascal_voc(gt_path)
    elif gt_type == 'txt':
        gt = parse_txt_groundtruth(gt_path)
    else:
        raise StandardError('Invalid detection type: {:s}'.format(detection_type))

    print 'detection_path: {:s}'.format(detection_path)
    if detection_type == 'json':
        detection = parse_json(detection_path)
    elif detection_type == 'txt':
        detection = parse_txt_detection(detection_path)
    else:
        raise StandardError('Invalid detection type: {:s}'.format(detection_type))

    rec, prec, scores = compute_pre_rec(gt=gt, detection=detection, ovthresh=ovthresh)
    print("Recall: {}".format(rec[-1]))
    print("Precision: {}".format(prec[-1]))
    print '{:f}\t{:f}'.format(rec[-1], prec[-1])
    # with open(args.output_path, 'w') as f:
    #     json.dump([rec, prec, scores], f)


