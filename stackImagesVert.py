import os
import shutil
import sys
import cv2

from datetime import datetime

from Misc import stackImages, add_suffix

image_paths = sys.argv[1:]
n_images = len(image_paths)

print('vertically stacking images {}'.format(image_paths))

src_images = [cv2.imread(image) for image in image_paths]
grid_size = [n_images, 1]


stacked_img, _, _ = stackImages(src_images, grid_size, borderless=1,
                                return_idx=1, preserve_order=1)

in_img_path = image_paths[0]
in_img_dir = os.path.dirname(in_img_path)
in_img_fname, in_img_ext = os.path.splitext(os.path.basename(in_img_path))

time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
# out_img_path = os.path.join(in_img_dir, in_img_fname + '_stacked_vert' + in_img_ext)
out_img_path = os.path.join(in_img_dir, time_stamp + in_img_ext)

print('saving stacked image to {}'.format(out_img_path))

cv2.imwrite(out_img_path, stacked_img)

dst_image_paths = [add_suffix(image, 'vert') for image in image_paths]
for src_path, dst_path in zip(image_paths, dst_image_paths):
    shutil.move(src_path, dst_path)
