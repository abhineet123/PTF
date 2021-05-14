import xml.etree.cElementTree as ET
import sys
import os
import glob
import numpy as np
import cv2

from pprint import pprint

import paramparse

from Misc import ParamDict, drawBox, resizeAR, imshow


def main():
    _params = {
        'root_dir': '/data',
        'actor_id': 5,
        'start_id': 1,
        'end_id': -1,
        'ignored_region_only': 0,
        'speed': 0.5,
        'show_img': 0,
        'quality': 3,
        'resize': 0,
        'mode': 0,
        'auto_progress': 0,
    }

    bkg_occl_seq = [24, 28, 47, 48, 49, 54]

    paramparse.process_dict(_params)

    root_dir = _params['root_dir']
    actor_id = _params['actor_id']
    start_id = _params['start_id']
    ignored_region_only = _params['ignored_region_only']
    end_id = _params['end_id']
    show_img = _params['show_img']

    params = ParamDict().__dict__
    actors = params['mot_actors']
    sequences = params['mot_sequences']

    actor = actors[actor_id]
    actor_sequences = sequences[actor]

    if end_id <= start_id:
        end_id = len(actor_sequences) - 1

    print('root_dir: {}'.format(root_dir))
    print('actor_id: {}'.format(actor_id))
    print('start_id: {}'.format(start_id))
    print('end_id: {}'.format(end_id))

    print('actor: {}'.format(actor))
    print('actor_sequences: {}'.format(actor_sequences))

    n_frames_list = []
    _pause = 1
    __pause = 1

    for seq_id in range(start_id, end_id + 1):
        seq_name = actor_sequences[seq_id]
        fname = '{:s}/{:s}/Annotations/xml/{:s}.xml'.format(root_dir, actor, seq_name)
        tree = ET.parse(fname)
        root = tree.getroot()

        out_seq_name = 'detrac_{}_{}'.format(seq_id + 1, seq_name)

        out_fname = '{:s}/{:s}/Annotations/{:s}.txt'.format(root_dir, actor, out_seq_name)
        out_fid = open(out_fname, 'w')

        ignored_region_obj = tree.find('ignored_region')
        n_ignored_regions = 0

        for bndbox in ignored_region_obj.iter('box'):
            if bndbox is None:
                continue
            xmin = float(bndbox.attrib['left'])
            ymin = float(bndbox.attrib['top'])
            width = float(bndbox.attrib['width'])
            height = float(bndbox.attrib['height'])
            out_fid.write('-1,-1,{:f},{:f},{:f},{:f},-1,-1,-1,-1,-1\n'.format(
                xmin, ymin, width, height))
            n_ignored_regions += 1

        if ignored_region_only:
            out_fid.close()
            continue

        img_dir = '{:s}/{:s}/Images/{:s}'.format(root_dir, actor, out_seq_name)
        frames = glob.glob('{:s}/*.jpg'.format(img_dir))
        n_frames = len(frames)
        n_frames_list.append(n_frames)
        seq_occluded_dict = {}

        skip_seq = 0

        print('Processing sequence {:d} :: {:s} n_ignored_regions: {}'.format(seq_id, seq_name, n_ignored_regions))
        for frame_obj in tree.iter('frame'):
            target_list = frame_obj.find('target_list')
            frame_id = int(frame_obj.attrib['num'])

            if show_img:
                frame_path = os.path.join(img_dir, 'image{:06d}.jpg'.format(frame_id))
                frame = cv2.imread(frame_path)
                obj_frame = np.copy(frame)

                if frame is None:
                    raise IOError('Failed to read frame: {}'.format(frame_path))
                cv2.putText(frame, str(frame_id), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 1, cv2.LINE_AA)

            occluded = []
            occluded_dict = {}
            obj_dict = {}

            for obj in target_list.iter('target'):
                bndbox = obj.find('box')
                obj_id = int(obj.attrib['id'])
                xmin = float(bndbox.attrib['left'])
                ymin = float(bndbox.attrib['top'])
                width = float(bndbox.attrib['width'])
                height = float(bndbox.attrib['height'])

                assert obj_id not in obj_dict, "duplicate object found"
                obj_dict[obj_id] = (xmin, ymin, width, height)

                for occ_idx, occ_obj in enumerate(obj.iter('occlusion')):
                    occlusion = occ_obj.find('region_overlap')
                    occ_status = int(occlusion.attrib['occlusion_status'])
                    occ_id = int(occlusion.attrib['occlusion_id'])
                    occ_xmin = float(occlusion.attrib['left'])
                    occ_ymin = float(occlusion.attrib['top'])
                    occ_width = float(occlusion.attrib['width'])
                    occ_height = float(occlusion.attrib['height'])
                    if occ_status == 0:
                        """occluded by another obj"""
                        _obj_id = obj_id
                        _occ_id = occ_id
                    elif occ_status == 1:
                        """occluding another obj"""
                        _obj_id = occ_id
                        _occ_id = obj_id
                    elif occ_status == -1:
                        """occluded by background"""
                        """"seems extremely unreliable so ignoring"""
                        # _obj_id = obj_id
                        # _occ_id = occ_id
                        continue
                    else:
                        raise AssertionError('Invalid occ_status: {}'.format(occ_status))

                    # assert _obj_id not in occluded_dict, "duplicate occlusion found"

                    if _obj_id not in occluded_dict:
                        occluded_dict[_obj_id] = []

                    occluded_dict[_obj_id].append((_occ_id, occ_status, occ_xmin, occ_ymin, occ_width, occ_height))

                    occluded.append((obj_id, occ_status, occ_id, occ_xmin, occ_ymin, occ_width, occ_height))

                    if occ_idx > 0:
                        raise AssertionError('Multiple occluding objects found')

            for obj_id in obj_dict:
                xmin, ymin, width, height = obj_dict[obj_id]
                xmax, ymax = xmin + width, ymin + height

                obj_img = np.zeros((int(height), int(width), 1), dtype=np.uint8)

                obj_img.fill(64)

                if obj_id in occluded_dict:
                    if show_img:
                        _obj_frame = np.copy(obj_frame)
                        drawBox(_obj_frame, xmin, ymin, xmax, ymax, label=str(obj_id),
                                box_color=(255, 255, 255))

                    # __pause = imshow('_obj_frame', _obj_frame, __pause)

                    for _occluded in occluded_dict[obj_id]:
                        occ_id, occ_status, occ_xmin, occ_ymin, occ_width, occ_height = _occluded
                        occ_xmax, occ_ymax = occ_xmin + occ_width, occ_ymin + occ_height

                        start_x, end_x = int(occ_xmin - xmin), int(occ_xmax - xmin)
                        start_y, end_y = int(occ_ymin - ymin), int(occ_ymax - ymin)

                        # assert start_x >= 0 and start_y >= 0, "Invalid occlusion region start: {}".format(_occluded)
                        # assert end_x <= width and end_y <= height, \
                        # "Invalid occlusion region end: {} for obj: {}\n{}, {}".format(
                        #     _occluded, obj_dict[obj_id], (end_x, end_y), (width, height))

                        start_x, start_y = max(start_x, 0), max(start_y, 0)
                        end_x, end_y = min(end_x, width), min(end_y, height)

                        obj_img[int(start_y):int(end_y), int(start_x):int(end_x)] = 192

                    n_occluded_pix = np.count_nonzero(obj_img == 192)
                    occluded_ratio = float(n_occluded_pix) / float(obj_img.size)

                    if show_img:
                        _obj_img = resizeAR(obj_img, 500)
                        _obj_frame = resizeAR(_obj_frame, 1920, 1080)
                        cv2.putText(_obj_img, str('{}-{} : {:.2f}'.format(obj_id, occ_id, occluded_ratio)), (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
                        __pause, k = imshow(('obj_img', '_obj_frame'), (_obj_img, _obj_frame), __pause)
                        if k == ord('q'):
                            print("skip_seq")
                            skip_seq = 1
                            break
                else:
                    occ_id = occ_xmin = occ_ymin = occ_width = occ_height = -1
                    occluded_ratio = 0

                out_str = '{:d},{:d},{:f},{:f},{:f},{:f},1,-1,-1,-1,{:f}\n'.format(
                    frame_id, obj_id, xmin, ymin, width, height, occluded_ratio)

                out_fid.write(out_str)

                seq_occluded_dict[frame_id] = occluded_dict

                if show_img:
                    drawBox(frame, xmin, ymin, xmax, ymax, label=str(obj_id),
                            box_color=(255, 255, 255), is_dotted=(occluded_ratio != 0))

            if skip_seq:
                break

            if show_img:
                for _occ in occluded:
                    obj_id, occ_status, occ_id, occ_xmin, occ_ymin, occ_width, occ_height = _occ

                    if occ_status == 1:
                        box_color = (0, 0, 255)
                    elif occ_status == 0:
                        box_color = (255, 0, 0)
                    elif occ_status == -1:
                        box_color = (0, 255, 0)

                    occ_xmax, occ_ymax = occ_xmin + occ_width, occ_ymin + occ_height
                    drawBox(frame, occ_xmin, occ_ymin, occ_xmax, occ_ymax,
                            box_color=box_color, thickness=1, label='{}-{}'.format(str(obj_id), str(occ_id)))

                frame = resizeAR(frame, 1920, 1080)
                _pause, k = imshow('frame', frame, _pause)

                if k == ord('q'):
                    print("skip_seq")
                    skip_seq = 1
                    break

            if frame_id % 100 == 0:
                print('\t Done {:d}/{:d} frames'.format(frame_id, n_frames))

        if skip_seq:
            continue

        meta_file_path = out_fname.replace('.txt', '.meta')
        with open(meta_file_path, 'w') as meta_fid:
            pprint(seq_occluded_dict, meta_fid)
        out_fid.close()

    print(n_frames_list)


if __name__ == '__main__':
    main()
