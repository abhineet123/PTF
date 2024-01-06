import os
import sys
import cv2

from pathlib import Path
from datetime import datetime

from Misc import stackImages
from Misc import processArguments

params = {
    'image_paths': [],
    'grid_size': '',
    'borderless': 1,
    'preserve_order': 1,
    'sep_size': 0,
}

processArguments(sys.argv[1:], params)
image_paths = params['image_paths']
grid_size = params['grid_size']
borderless = params['borderless']
preserve_order = params['preserve_order']
sep_size = params['sep_size']

if len(image_paths) == 1 and os.path.isdir(image_paths[0]):
    image_paths = [str(k) for k in Path(image_paths[0]).rglob('*.jpg')]

n_images = len(image_paths)

if not grid_size:
    grid_size = None
else:
    grid_size = [int(x) for x in grid_size.split('x')]
    assert len(grid_size) == 2, f"invalid grid_size: {grid_size}"
    assert grid_size[0] * grid_size[1] >= n_images, f"invalid grid_size {grid_size} for {n_images} images"

print('vertically stacking images {}'.format(image_paths))

src_images = [cv2.imread(image) for image in image_paths]

stacked_img, _, _ = stackImages(src_images, grid_size, borderless=borderless,
                                return_idx=1, preserve_order=preserve_order, sep_size=sep_size)

in_img_path = image_paths[0]
in_img_dir = os.path.dirname(in_img_path)
in_img_fname, in_img_ext = os.path.splitext(os.path.basename(in_img_path))

time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
# out_img_path = os.path.join(in_img_dir, in_img_fname + '_stacked_vert' + in_img_ext)
out_img_path = os.path.join(in_img_dir, time_stamp + in_img_ext)

print('saving stacked image to {}'.format(out_img_path))

cv2.imwrite(out_img_path, stacked_img)
