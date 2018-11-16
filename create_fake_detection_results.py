from utils.annotation_parsing import parse_txt_groundtruth, write_MOT_detection, convert_ground_truth_to_detection
import argparse
import numpy as np
from PIL import Image

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str)
  parser.add_argument('output', type=str)
  parser.add_argument('recall', type=float)
  parser.add_argument('precision', type=float)
  parser.add_argument('image_width', type=int)
  parser.add_argument('image_height', type=int)
  parser.add_argument('max_frame_id', type=int)
  parser.add_argument('--roi_path', default=None, type=str)
  return parser.parse_args()

def get_translate(current, max, translate):
  if current + translate < 1:
    return current - 1
  if current + translate > max:
    return max - current
  return translate

def random_translate(bbox, im_width, im_height, translate_percent=0.15):
  bbox_w = bbox[2] - bbox[0] + 1
  bbox_h = bbox[3] - bbox[1] + 1
  translate_x = int(translate_percent * bbox_w)
  translate_x = np.random.randint(- translate_x, translate_x + 1)
  translate_x = get_translate(bbox[0], im_width, translate_x)
  translate_y = int(translate_percent * bbox_h)
  translate_y = np.random.randint(- translate_y, translate_y + 1)
  translate_y = get_translate(bbox[1], im_height, translate_y)
  return [bbox[0] + translate_x,
          bbox[1] + translate_y,
          bbox[2] + translate_x,
          bbox[3] + translate_y]

def create_false_postives(im_width, im_height, max_frame_id, num_frame=1):
  frames = np.zeros((num_frame, 7))
  frames[:, 2] = np.random.randint(1, im_width - 4)
  frames[:, 3] = np.random.randint(1, im_height - 4)
  frames[:, 4] = np.random.randint(5, int(im_width / 2))
  frames[:, 5] = np.random.randint(5, int(im_height / 2))
  frames[:, 4] = frames[:, 2] + frames[:, 4]
  frames[:, 5] = frames[:, 3] + frames[:, 5]
  frames[:, 4][frames[:, 4] > im_width] = im_width
  frames[:, 5][frames[:, 5] > im_height] = im_height
  frames[:, 0] = np.random.randint(1, max_frame_id + 1, size=[num_frame])
  frames[:, 1] = -1
  frames[:, 6] = 0.1
  return frames

def in_roi_check(roi, bbox):
  height, width = roi.shape
  y_min = max(int(bbox[1]) - 1, 0)
  y_max = min(int(bbox[3]) - 1, height)
  x_min = max(int(bbox[0]) - 1, 0)
  x_max = min(int(bbox[2]) - 1, width)
  _bbox = np.zeros_like(roi)
  _bbox[y_min:y_max + 1, x_min:x_max + 1] = 1
  return (_bbox * roi).sum() > 0

def read_roi(fn):
  roi = np.asarray(Image.open(fn), dtype=np.int32)
  roi = roi.sum(axis=2)
  roi.setflags(write=1)
  roi[roi <= 100] = 0
  roi[roi > 100] = 1
  return roi

if __name__ == '__main__':
  args = parse_args()
  if args.roi_path is not None:
    roi = read_roi(args.roi_path)
  else:
    roi = None
  true_bboxes = parse_txt_groundtruth(args.input)
  true_bboxes = convert_ground_truth_to_detection(true_bboxes)
  sorted_ids = np.argsort(true_bboxes[:, 0])
  true_bboxes = true_bboxes[sorted_ids, :]
  # delete
  bboxes_delete_scores = np.random.rand(len(true_bboxes))
  sorted_ids = np.argsort(bboxes_delete_scores)
  delete_ids = sorted_ids[0: int((1 - args.recall) * len(true_bboxes))]
  true_bboxes = np.delete(true_bboxes, delete_ids, axis=0)
  # translate bboxes randomly
  for bbox in true_bboxes:
    bbox[2:6] = random_translate(bbox[2:6], args.image_width, args.image_height, 0.15)
  # create false positive
  false_positive_num = int((1 / args.precision - 1) * len(true_bboxes))
  false_bboxes = []
  if roi is not None:
    for _ in range(false_positive_num):
      while True:
        bbox = list(*create_false_postives(args.image_width, args.image_height, args.max_frame_id))
        if in_roi_check(roi, bbox):
          break
      false_bboxes.append(bbox)
    false_bboxes = np.vstack(false_bboxes)
  else:
    false_bboxes = create_false_postives(args.image_width, args.image_height, args.max_frame_id, false_positive_num)
  all_bboxes = np.vstack([true_bboxes, false_bboxes])
  sorted_ids = np.argsort(all_bboxes[:, 0])
  all_bboxes = all_bboxes[sorted_ids, :]
  write_MOT_detection(args.output, all_bboxes, 0.0)
