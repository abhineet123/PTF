__author__ = 'Tommy'
import os
import sys
import re
import shutil

try:
    from PIL import Image, ImageChops
except ImportError as e:
    print('PIL import failed: {}'.format(e))

try:
    import cv2
except ImportError as e:
    print('OpenCV import failed: {}'.format(e))

try:
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib import font_manager as ftm
    # from matplotlib.mlab import PCA
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError as e:
    print('matplotlib import failed: {}'.format(e))

import copy
import struct
import time
from tqdm import tqdm

# from DecompUtils import getBinaryPtsImage2
from Homography import *

# import six

col_rgb = {
    'snow': (250, 250, 255),
    'snow_2': (233, 233, 238),
    'snow_3': (201, 201, 205),
    'snow_4': (137, 137, 139),
    'ghost_white': (255, 248, 248),
    'white_smoke': (245, 245, 245),
    'gainsboro': (220, 220, 220),
    'floral_white': (240, 250, 255),
    'old_lace': (230, 245, 253),
    'linen': (230, 240, 240),
    'antique_white': (215, 235, 250),
    'antique_white_2': (204, 223, 238),
    'antique_white_3': (176, 192, 205),
    'antique_white_4': (120, 131, 139),
    'papaya_whip': (213, 239, 255),
    'blanched_almond': (205, 235, 255),
    'bisque': (196, 228, 255),
    'bisque_2': (183, 213, 238),
    'bisque_3': (158, 183, 205),
    'bisque_4': (107, 125, 139),
    'peach_puff': (185, 218, 255),
    'peach_puff_2': (173, 203, 238),
    'peach_puff_3': (149, 175, 205),
    'peach_puff_4': (101, 119, 139),
    'navajo_white': (173, 222, 255),
    'moccasin': (181, 228, 255),
    'cornsilk': (220, 248, 255),
    'cornsilk_2': (205, 232, 238),
    'cornsilk_3': (177, 200, 205),
    'cornsilk_4': (120, 136, 139),
    'ivory': (240, 255, 255),
    'ivory_2': (224, 238, 238),
    'ivory_3': (193, 205, 205),
    'ivory_4': (131, 139, 139),
    'lemon_chiffon': (205, 250, 255),
    'seashell': (238, 245, 255),
    'seashell_2': (222, 229, 238),
    'seashell_3': (191, 197, 205),
    'seashell_4': (130, 134, 139),
    'honeydew': (240, 255, 240),
    'honeydew_2': (224, 238, 244),
    'honeydew_3': (193, 205, 193),
    'honeydew_4': (131, 139, 131),
    'mint_cream': (250, 255, 245),
    'azure': (255, 255, 240),
    'alice_blue': (255, 248, 240),
    'lavender': (250, 230, 230),
    'lavender_blush': (245, 240, 255),
    'misty_rose': (225, 228, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'dark_slate_gray': (79, 79, 49),
    'dim_gray': (105, 105, 105),
    'slate_gray': (144, 138, 112),
    'light_slate_gray': (153, 136, 119),
    'gray': (190, 190, 190),
    'light_gray': (211, 211, 211),
    'midnight_blue': (112, 25, 25),
    'navy': (128, 0, 0),
    'cornflower_blue': (237, 149, 100),
    'dark_slate_blue': (139, 61, 72),
    'slate_blue': (205, 90, 106),
    'medium_slate_blue': (238, 104, 123),
    'light_slate_blue': (255, 112, 132),
    'medium_blue': (205, 0, 0),
    'royal_blue': (225, 105, 65),
    'blue': (255, 0, 0),
    'dodger_blue': (255, 144, 30),
    'deep_sky_blue': (255, 191, 0),
    'sky_blue': (250, 206, 135),
    'light_sky_blue': (250, 206, 135),
    'steel_blue': (180, 130, 70),
    'light_steel_blue': (222, 196, 176),
    'light_blue': (230, 216, 173),
    'powder_blue': (230, 224, 176),
    'pale_turquoise': (238, 238, 175),
    'dark_turquoise': (209, 206, 0),
    'medium_turquoise': (204, 209, 72),
    'turquoise': (208, 224, 64),
    'cyan': (255, 255, 0),
    'light_cyan': (255, 255, 224),
    'cadet_blue': (160, 158, 95),
    'medium_aquamarine': (170, 205, 102),
    'aquamarine': (212, 255, 127),
    'dark_green': (0, 100, 0),
    'dark_olive_green': (47, 107, 85),
    'dark_sea_green': (143, 188, 143),
    'sea_green': (87, 139, 46),
    'medium_sea_green': (113, 179, 60),
    'light_sea_green': (170, 178, 32),
    'pale_green': (152, 251, 152),
    'spring_green': (127, 255, 0),
    'lawn_green': (0, 252, 124),
    'chartreuse': (0, 255, 127),
    'medium_spring_green': (154, 250, 0),
    'green_yellow': (47, 255, 173),
    'lime_green': (50, 205, 50),
    'yellow_green': (50, 205, 154),
    'forest_green': (34, 139, 34),
    'olive_drab': (35, 142, 107),
    'dark_khaki': (107, 183, 189),
    'khaki': (140, 230, 240),
    'pale_goldenrod': (170, 232, 238),
    'light_goldenrod_yellow': (210, 250, 250),
    'light_yellow': (224, 255, 255),
    'yellow': (0, 255, 255),
    'gold': (0, 215, 255),
    'light_goldenrod': (130, 221, 238),
    'goldenrod': (32, 165, 218),
    'dark_goldenrod': (11, 134, 184),
    'rosy_brown': (143, 143, 188),
    'indian_red': (92, 92, 205),
    'saddle_brown': (19, 69, 139),
    'sienna': (45, 82, 160),
    'peru': (63, 133, 205),
    'burlywood': (135, 184, 222),
    'beige': (220, 245, 245),
    'wheat': (179, 222, 245),
    'sandy_brown': (96, 164, 244),
    'tan': (140, 180, 210),
    'chocolate': (30, 105, 210),
    'firebrick': (34, 34, 178),
    'brown': (42, 42, 165),
    'dark_salmon': (122, 150, 233),
    'salmon': (114, 128, 250),
    'light_salmon': (122, 160, 255),
    'orange': (0, 165, 255),
    'dark_orange': (0, 140, 255),
    'coral': (80, 127, 255),
    'light_coral': (128, 128, 240),
    'tomato': (71, 99, 255),
    'orange_red': (0, 69, 255),
    'red': (0, 0, 255),
    'hot_pink': (180, 105, 255),
    'deep_pink': (147, 20, 255),
    'pink': (203, 192, 255),
    'light_pink': (193, 182, 255),
    'pale_violet_red': (147, 112, 219),
    'maroon': (96, 48, 176),
    'medium_violet_red': (133, 21, 199),
    'violet_red': (144, 32, 208),
    'violet': (238, 130, 238),
    'plum': (221, 160, 221),
    'orchid': (214, 112, 218),
    'medium_orchid': (211, 85, 186),
    'dark_orchid': (204, 50, 153),
    'dark_violet': (211, 0, 148),
    'blue_violet': (226, 43, 138),
    'purple': (240, 32, 160),
    'medium_purple': (219, 112, 147),
    'thistle': (216, 191, 216),
    'green': (0, 255, 0),
    'magenta': (255, 0, 255)
}


def getBinaryPtsImage2(img_shape, corners):
    img_shape = img_shape[0:2]
    corners = corners.transpose().astype(np.int32)
    # print 'corners:\n ', corners
    bin_img = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillConvexPoly(bin_img, corners, (255, 255, 255))

    # drawRegion(bin_img, corners, (255, 255, 255), thickness=1)
    # # print 'img_shape: ', img_shape
    # min_row = max(0, int(np.amin(corners[1, :])))
    # max_row = min(img_shape[0], int(np.amax(corners[1, :])))
    # for row in xrange(min_row, max_row):
    # non_zero_idx = np.transpose(np.nonzero(bin_img[row, :]))
    # # print 'curr_row: ', curr_row
    # # print 'non_zero_idx: ', non_zero_idx
    # bin_img[row, non_zero_idx[0]:non_zero_idx[-1]] = 255
    return bin_img


# def getBinaryPtsImage2(img_shape, corners):
# img_shape = img_shape[0:2]
# bin_img = np.zeros(img_shape, dtype=np.uint8)
# drawRegion(bin_img, corners, (255, 255, 255), thickness=1)
# # print 'img_shape: ', img_shape
# for row in xrange(img_shape[0]):
# non_zero_idx = np.transpose(np.nonzero(bin_img[row, :]))
# # print 'curr_row: ', curr_row
# # print 'non_zero_idx: ', non_zero_idx
# if non_zero_idx.shape[0]>1:
# bin_img[row, non_zero_idx[0]:non_zero_idx[-1]] = 255
# return bin_img

def sortKey(fname, only_basename=1):
    if only_basename:
        fname = os.path.splitext(os.path.basename(fname))[0]
    else:
        # remove extension
        fname = os.path.join(os.path.dirname(fname),
                             os.path.splitext(os.path.basename(fname))[0])
        fname_list = fname.split(os.sep)
        # print('fname_list: ', fname_list)
        out_name = ''
        for _dir in fname_list:
            out_name = '{}_{}'.format(out_name, _dir) if out_name else _dir
        # print('zip_path: ', out_name)
        fname = out_name
    # print('fname: ', fname)
    # split_fname = fname.split('_')
    # print('split_fname: ', split_fname)

    split_list = fname.split('_')
    key = ''

    for s in split_list:
        if s.isdigit():
            if not key:
                key = '{:012d}'.format(int(s))
            else:
                key = '{}-{:012d}'.format(key, int(s))
        else:
            try:
                nums = [int(k) for k in re.findall(r'\d+', s)]
                # print('\ts: {}\t nums: {}\n'.format(s, nums))
                for num in nums:
                    s = s.replace(str(num), '{:012d}'.format(num))
            except ValueError:
                pass
            if not key:
                key = s
            else:
                key = '{}-{}'.format(key, s)
    # print('fname: {}\t key: {}\n'.format(fname, key))
    return key


def sortKeyOld(fname):
    fname = os.path.splitext(fname)[0]
    # print('fname: ', fname)
    # split_fname = fname.split('_')
    # print('split_fname: ', split_fname)
    nums = [int(s) for s in fname.split('_') if s.isdigit()]
    non_nums = [s for s in fname.split('_') if not s.isdigit()]
    key = ''
    for non_num in non_nums:
        if not key:
            key = non_num
        else:
            key = '{}_{}'.format(key, non_num)
    for num in nums:
        if not key:
            key = '{:08d}'.format(num)
        else:
            key = '{}_{:08d}'.format(key, num)

    # try:
    #     key = nums[-1]
    # except IndexError:
    #     return fname

    # print('key: ', key)
    return key


def trim(im, all_corners=0, margin=-5):
    im_size = im.size
    # print('im_size: {}'.format(im_size))

    h, w = im_size[:2]
    bg_pix_ul = im.getpixel((0, 0))

    bg_pixs = [bg_pix_ul, ]

    if all_corners:
        bg_pix_ur = im.getpixel((0, w - 1))
        bg_pix_ll = im.getpixel((h - 1, 0))
        bg_pix_lr = im.getpixel((h - 1, w - 1))

        bg_pixs += [bg_pix_ur, bg_pix_ll, bg_pix_lr]
    # im.show()
    # print('bg_pix: {}'.format(bg_pix))

    for pix in bg_pixs:
        bg = Image.new(im.mode, im_size, pix)
        diff = ImageChops.difference(im, bg)
        # diff.show()
        diff = ImageChops.add(diff, diff, 2.0, margin)
        # diff.show()
        # diff = ImageChops.add(diff, diff)
        bbox = diff.getbbox()
        if bbox:
            im = im.crop(bbox)
    return im


def move_or_del_files(src_path, filenames, dst_path='', remove_empty=1):
    if dst_path:
        print('moving files from {} --> {}'.format(src_path, dst_path))
        os.makedirs(dst_path, exist_ok=True)
    else:
        print('deleting files in {}'.format(src_path))

    for filename in tqdm(filenames):
        file_src_path = os.path.join(src_path, filename)
        if not dst_path:
            os.remove(file_src_path)
        else:
            file_dst_path = os.path.join(dst_path, filename)
            shutil.move(file_src_path, file_dst_path)

    if remove_empty and not os.listdir(src_path):
        print('deleting empty folder: {}'.format(src_path))
        shutil.rmtree(src_path)


def processArguments(args, params):
    # arguments specified as 'arg_name=argv_val'
    no_of_args = len(args)
    for arg_id in range(no_of_args):
        arg_str = args[arg_id]

        if arg_str.startswith('--'):
            arg_str = arg_str[2:]

        arg = arg_str.split('=')[-2:]

        if len(arg) == 3:
            arg = arg[1:]

        # print('args[{}]: {}'.format(arg_id, args[arg_id]))
        # print('arg: {}'.format(arg))
        if len(arg) != 2 or arg[0] not in params.keys():
            raise IOError('Invalid argument provided: {:s}'.format(args[arg_id]))
            return

        if not arg[1] or not arg[0] or arg[1] == '#':
            continue

        if isinstance(params[arg[0]], (list, tuple)):
            # if not ',' in arg[1]:
            #     print('Invalid argument provided for list: {:s}'.format(arg[1]))
            #     return

            if arg[1] and ',' not in arg[1]:
                arg[1] = '{},'.format(arg[1])

            arg_vals = arg[1].split(',')
            arg_vals_parsed = []
            for _val in arg_vals:
                try:
                    _val_parsed = int(_val)
                except ValueError:
                    try:
                        _val_parsed = float(_val)
                    except ValueError:
                        _val_parsed = _val if _val else None
                        if _val_parsed == '__n__':
                            _val_parsed = ''

                if _val_parsed is not None:
                    arg_vals_parsed.append(_val_parsed)
            params[arg[0]] = arg_vals_parsed
        else:
            params[arg[0]] = type(params[arg[0]])(arg[1])
            if isinstance(params[arg[0]], str) and params[arg[0]] == '__n__':
                params[arg[0]] = ''


def resizeAR(src_img, width=0, height=0, return_factors=False,
             placement_type=0, resize_factor=0):
    src_height, src_width, n_channels = src_img.shape

    src_aspect_ratio = float(src_width) / float(src_height)

    if isinstance(placement_type, int):
        placement_type = (placement_type, placement_type)

    # print('placement_type: {}'.format(placement_type))

    # print('placement_type: {}'.format(placement_type))

    if resize_factor != 0:
        width, height = int(src_width * resize_factor), int(src_height * resize_factor)

    if width <= 0 and height <= 0:
        if resize_factor == 0:
            raise AssertionError(
                'Both width and height cannot be 0 when resize_factor is 0 too')
    elif height <= 0:
        height = int(width / src_aspect_ratio)
    elif width <= 0:
        width = int(height * src_aspect_ratio)

    aspect_ratio = float(width) / float(height)

    # print('src_aspect_ratio: {}'.format(src_aspect_ratio))
    # print('aspect_ratio: {}'.format(aspect_ratio))

    if src_aspect_ratio == aspect_ratio:
        dst_width = src_width
        dst_height = src_height
        start_row = start_col = 0
    elif src_aspect_ratio > aspect_ratio:
        dst_width = src_width
        dst_height = int(src_width / aspect_ratio)
        start_row = int((dst_height - src_height) / 2.0)
        if placement_type[0] == 0:
            start_row = 0
            # start_row = int(dst_height - src_height)
        elif placement_type[0] == 1:
            start_row = int((dst_height - src_height) / 2.0)
        elif placement_type[0] == 2:
            # start_row = 0
            start_row = int(dst_height - src_height)
        start_col = 0
    else:
        dst_height = src_height
        dst_width = int(src_height * aspect_ratio)
        start_col = int((dst_width - src_width) / 2.0)
        if placement_type[1] == 0:
            start_col = 0
        elif placement_type[1] == 1:
            start_col = int((dst_width - src_width) / 2.0)
        elif placement_type[1] == 2:
            start_col = int(dst_width - src_width)
        start_row = 0

    dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=np.uint8)

    dst_img[start_row:start_row + src_height, start_col:start_col + src_width, :] = src_img
    dst_img = cv2.resize(dst_img, (width, height))
    if return_factors:
        resize_factor = float(height) / float(dst_height)
        return dst_img, resize_factor, start_row, start_col
    else:
        return dst_img


def sizeAR(src_img, height=0, width=0):
    src_height, src_width, n_channels = src_img.shape

    src_aspect_ratio = float(src_width) / float(src_height)

    if width <= 0 and height <= 0:
        return src_height, src_width
    elif height <= 0:
        height = int(width / src_aspect_ratio)
    elif width <= 0:
        width = int(height * src_aspect_ratio)

    return height, width


def str2num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def print_and_write(_str, fname=None):
    sys.stdout.write(_str + '\n')
    sys.stdout.flush()
    if fname is not None:
        open(fname, 'a').write(_str + '\n')


def printMatrixToFile(mat, mat_name, fname, fmt='{:15.9f}', mode='w', sep='\t'):
    fid = open(fname, mode)
    fid.write('{:s}:\n'.format(mat_name))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            fid.write(fmt.format(mat[i, j]) + sep)
        fid.write('\n')
    fid.write('\n\n')
    fid.close()


def printVectorFile(mat, mat_name, fname, fmt='{:15.9f}', mode='w', sep='\t'):
    fid = open(fname, mode)
    # print 'mat before: ', mat
    mat = mat.squeeze()
    # mat = mat.squeeze()
    # if len(mat.shape) > 1:
    # print 'here we are outside'
    # mat = mat[0, 0]
    #
    # print 'mat: ', mat
    # print 'mat.size: ', mat.size
    # print 'mat.shape: ', mat.shape
    fid.write('{:s}:\n'.format(mat_name))
    for i in range(mat.size):
        val = mat[i]
        # print 'type( val ): ', type( val )
        # if not isinstance(val, (int, long, float)) or type( val )=='numpy.float64':
        # print 'here we are'
        # val = mat[0, i]
        # print 'val: ', val
        fid.write(fmt.format(val) + sep)
    fid.write('\n')
    fid.write('\n\n')
    fid.close()


def printScalarToFile(scalar_val, scalar_name,
                      fname, fmt='{:15.9f}', mode='w'):
    fid = open(fname, mode)
    fid.write('{:s}:\t'.format(scalar_name))
    fid.write(fmt.format(scalar_val))
    fid.write('\n\n')
    fid.close()


def getTrackingObject(img, col=(0, 0, 255), title=None):
    annotated_img = img.copy()
    temp_img = img.copy()
    if title is None:
        title = 'Select the object to track'
    cv2.namedWindow(title)
    cv2.imshow(title, annotated_img)
    pts = []

    def drawLines(img, hover_pt=None):
        if len(pts) == 0:
            cv2.imshow(title, img)
            return
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], col, 1)
        if hover_pt is None:
            return
        cv2.line(img, pts[-1], hover_pt, col, 1)
        if len(pts) == 3:
            cv2.line(img, pts[0], hover_pt, col, 1)
        elif len(pts) == 4:
            return
        cv2.imshow(title, img)

    def mouseHandler(event, x, y, flags=None, param=None):
        if len(pts) >= 4:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            temp_img = annotated_img.copy()
            drawLines(temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            pass
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(pts) > 0:
                print('Removing last point')
                del (pts[-1])
            temp_img = annotated_img.copy()
            drawLines(temp_img)
        elif event == cv2.EVENT_RBUTTONUP:
            pass
        elif event == cv2.EVENT_MBUTTONDOWN:
            pass
        elif event == cv2.EVENT_MOUSEMOVE:
            # if len(pts) == 0:
            # return
            temp_img = annotated_img.copy()
            drawLines(temp_img, (x, y))

    cv2.setMouseCallback(title, mouseHandler, param=[annotated_img, temp_img, pts])
    while len(pts) < 4:
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.waitKey(250)
    cv2.destroyWindow(title)
    drawLines(annotated_img, pts[0])
    return pts, annotated_img


def getTrackingObject2(img, col=(0, 0, 255), title=None, line_thickness=1):
    annotated_img = img.copy()
    temp_img = img.copy()
    if title is None:
        title = 'Select the object to track'
    cv2.namedWindow(title)
    cv2.imshow(title, annotated_img)
    pts = []

    def drawLines(img, hover_pt=None):
        if len(pts) == 0:
            cv2.imshow(title, img)
            return
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], col, line_thickness)
        if hover_pt is None:
            return
        cv2.line(img, pts[-1], hover_pt, col, line_thickness)
        if len(pts) == 3:
            cv2.line(img, pts[0], hover_pt, col, line_thickness)
        elif len(pts) == 4:
            return
        cv2.imshow(title, img)

    def mouseHandler(event, x, y, flags=None, param=None):
        if len(pts) >= 4:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            temp_img = annotated_img.copy()
            drawLines(temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            pass
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(pts) > 0:
                print('Removing last point')
                del (pts[-1])
            temp_img = annotated_img.copy()
            drawLines(temp_img)
        elif event == cv2.EVENT_RBUTTONUP:
            pass
        elif event == cv2.EVENT_MBUTTONDOWN:
            pass
        elif event == cv2.EVENT_MOUSEMOVE:
            # if len(pts) == 0:
            # return
            temp_img = annotated_img.copy()
            drawLines(temp_img, (x, y))

    cv2.setMouseCallback(title, mouseHandler, param=[annotated_img, temp_img, pts])
    while len(pts) < 4:
        key = cv2.waitKey(1)
        if key == 27:
            sys.exit()
    cv2.waitKey(250)
    cv2.destroyWindow(title)
    drawLines(annotated_img, pts[0])
    return pts


def readTrackingData(filename, arch_fid=None):
    if arch_fid is not None:
        data_file = arch_fid.open(filename, 'r')
    else:
        if not os.path.isfile(filename):
            print("Tracking data file not found:\n ", filename)
            return None
        data_file = open(filename, 'r')

    data_file.readline()
    lines = data_file.readlines()
    data_file.close()
    no_of_lines = len(lines)
    data_array = np.empty([no_of_lines, 8])
    line_id = 0
    for line in lines:
        # print(line)
        words = line.split()
        coordinates = []
        if len(words) != 9:
            if len(words) == 2 and words[1] == 'invalid_tracker_state':
                for i in range(8):
                    coordinates.append(float('NaN'))
            else:
                msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
                raise SyntaxError(msg)
        else:
            words = words[1:]
            for word in words:
                coordinates.append(float(word))
        data_array[line_id, :] = coordinates
        # print words
        line_id += 1

    return data_array


def getFileList(root_dir, ext):
    file_list = []
    for file in os.listdir(root_dir):
        if file.endswith(ext):
            file_list.append(os.path.join(root_dir, file))
    return file_list


def readTrackingDataMOT(filename, arch_fid=None):
    if arch_fid is not None:
        data_file = arch_fid.open(filename, 'r')
    else:
        if not os.path.isfile(filename):
            print("Tracking data file not found:\n ", filename)
            return None
        data_file = open(filename, 'r')
    lines = data_file.readlines()
    data_file.close()
    no_of_lines = len(lines)
    data_array = np.empty([no_of_lines, 10])
    line_id = 0
    n_frames = 0
    for line in lines:
        # print(line)
        words = line.split(',')
        data = []
        if len(words) != 10:
            if len(words) == 7:
                for i in range(3):
                    words.append('-1')
            else:
                msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
                raise SyntaxError(msg)
        for word in words:
            data.append(float(word))
        data_array[line_id, :] = data
        # print words
        line_id += 1

    return data_array


# new version that supports reinit data as well as invalid tracker states
def readTrackingData2(tracker_path, n_frames, _arch_fid=None, _reinit_from_gt=0):
    print('Reading tracking data from: {:s}...'.format(tracker_path))
    if _arch_fid is not None:
        tracking_data = _arch_fid.open(tracker_path, 'r').readlines()
    else:
        tracking_data = open(tracker_path, 'r').readlines()
    if len(tracking_data) < 2:
        print('Tracking data file is invalid.')
        return None, None

    # remove the header
    del (tracking_data[0])
    n_lines = len(tracking_data)

    if not _reinit_from_gt and n_lines != n_frames:
        print("No. of frames in tracking result ({:d}) and the ground truth ({:d}) do not match".format(
            n_lines, n_frames))
        return None
    line_id = 1
    failure_count = 0
    invalid_tracker_state_found = False
    data_array = []
    while line_id < n_lines:
        tracking_data_line = tracking_data[line_id].strip().split()
        frame_fname = str(tracking_data_line[0])
        fname_len = len(frame_fname)
        frame_fname_1 = frame_fname[0:5]
        frame_fname_2 = frame_fname[- 4:]
        if frame_fname_1 != 'frame' or frame_fname_2 != '.jpg':
            print('Invaid formatting on tracking data line {:d}: {:s}'.format(line_id + 1, tracking_data_line))
            print('frame_fname: {:s} fname_len: {:d} frame_fname_1: {:s} frame_fname_2: {:s}'.format(
                frame_fname, fname_len, frame_fname_1, frame_fname_2))
            return None, None
        frame_id_str = frame_fname[5:-4]
        frame_num = int(frame_id_str)
        if len(tracking_data_line) != 9:
            if _reinit_from_gt and len(tracking_data_line) == 2 and tracking_data_line[1] == 'tracker_failed':
                print('tracking failure detected in frame: {:d} at line {:d}'.format(frame_num, line_id + 1))
                failure_count += 1
                data_array.append('tracker_failed')
                line_id += 2
                continue
            elif len(tracking_data_line) == 2 and tracking_data_line[1] == 'invalid_tracker_state':
                if not invalid_tracker_state_found:
                    print('invalid tracker state detected in frame: {:d} at line {:d}'.format(frame_num, line_id + 1))
                    invalid_tracker_state_found = True
                line_id += 1
                data_array.append('invalid_tracker_state')
                continue
            else:
                print('Invalid formatting on line {:d}: {:s}'.format(line_id, tracking_data[line_id]))
                return None, None
        data_array.append([float(tracking_data_line[1]), float(tracking_data_line[2]),
                           float(tracking_data_line[3]), float(tracking_data_line[4]),
                           float(tracking_data_line[5]), float(tracking_data_line[6]),
                           float(tracking_data_line[7]), float(tracking_data_line[8])])
    return data_array, failure_count


def readWarpData(filename):
    if not os.path.isfile(filename):
        print("Warp data file not found:\n ", filename)
        sys.exit()

    data_file = open(filename, 'r')
    lines = data_file.readlines()
    lines = (line.rstrip() for line in lines)
    lines = list(line for line in lines if line)
    data_file.close()
    no_of_lines = len(lines)
    warps_array = np.zeros((no_of_lines, 9), dtype=np.float64)
    line_id = 0
    for line in lines:
        words = line.split()
        if len(words) != 9:
            msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
            raise SyntaxError(msg)
        curr_warp = []
        for word in words:
            curr_warp.append(float(word))
        warps_array[line_id, :] = curr_warp
        line_id += 1
    return warps_array


def getNormalizedUnitSquarePts(resx=100, resy=100, c=1.0):
    pts_arr = np.mat(np.zeros((2, resy * resx)))
    pt_id = 0
    for y in np.linspace(-c, c, resy):
        for x in np.linspace(-c, c, resx):
            pts_arr[0, pt_id] = x
            pts_arr[1, pt_id] = y
            pt_id += 1
    corners = np.mat([[-c, c, c, -c], [-c, -c, c, c]])
    return pts_arr, corners


def drawRegion(img, corners, color, thickness=1, annotate_corners=False,
               annotation_col=(0, 255, 0), annotation_font_size=1):
    n_pts = corners.shape[1]
    for i in range(n_pts):
        p1 = (int(corners[0, i]), int(corners[1, i]))
        p2 = (int(corners[0, (i + 1) % n_pts]), int(corners[1, (i + 1) % n_pts]))
        if cv2.__version__.startswith('21'):
            cv2.line(img, p1, p2, color, thickness, cv2.CV_AA)
        else:
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
        if annotate_corners:
            if annotation_col is None:
                annotation_col = color
            cv2.putText(img, '{:d}'.format(i + 1), p1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        annotation_font_size, annotation_col)


def getPixValsRGB(pts, img):
    try:
        n_channels = img.shape[2]
    except IndexError:
        n_channels = 1
    # print 'img: ', img
    n_pts = pts.shape[1]
    pix_vals = np.zeros((n_pts, n_channels), dtype=np.float64)
    for channel_id in range(n_channels):
        try:
            curr_channel = img[:, :, channel_id].astype(np.float64)
        except IndexError:
            curr_channel = img
        pix_vals[:, channel_id] = getPixVals(pts, curr_channel)
    return pix_vals


def getPixVals(pts, img):
    x = pts[0, :]
    y = pts[1, :]

    n_rows, n_cols = img.shape
    x[x < 0] = 0
    y[y < 0] = 0
    x[x > n_cols - 1] = n_cols - 1
    y[y > n_rows - 1] = n_rows - 1

    lx = np.floor(x).astype(np.uint16)
    ux = np.ceil(x).astype(np.uint16)
    ly = np.floor(y).astype(np.uint16)
    uy = np.ceil(y).astype(np.uint16)

    dx = x - lx
    dy = y - ly

    ll = np.multiply((1 - dx), (1 - dy))
    lu = np.multiply(dx, (1 - dy))
    ul = np.multiply((1 - dx), dy)
    uu = np.multiply(dx, dy)

    # n_rows, n_cols = img.shape
    # lx[lx < 0] = 0
    # lx[lx >= n_cols] = n_cols - 1
    # ux[ux < 0] = 0
    # ly[ly < 0] = 0
    # ly[ly >= n_rows] = n_rows - 1
    # uy[uy < 0] = 0
    # ux[ux >= n_cols] = n_cols - 1
    # uy[uy >= n_rows] = n_rows - 1

    return np.multiply(img[ly, lx], ll) + np.multiply(img[ly, ux], lu) + \
           np.multiply(img[uy, lx], ul) + np.multiply(img[uy, ux], uu)


def drawGrid(img, pts, res_x, res_y, color, thickness=1):
    # draw vertical lines
    for x_id in range(res_x):
        for y_id in range(res_y - 1):
            pt_id1 = y_id * res_x + x_id
            pt_id2 = (y_id + 1) * res_x + x_id
            p1 = (int(pts[0, pt_id1]), int(pts[1, pt_id1]))
            p2 = (int(pts[0, pt_id2]), int(pts[1, pt_id2]))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    # draw horizontal lines
    for y_id in range(res_y):
        for x_id in range(res_x - 1):
            pt_id1 = y_id * res_x + x_id
            pt_id2 = y_id * res_x + x_id + 1
            p1 = (int(pts[0, pt_id1]), int(pts[1, pt_id1]))
            p2 = (int(pts[0, pt_id2]), int(pts[1, pt_id2]))
            cv2.line(img, p1, p2, color, thickness)


def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=7):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    # if style == 'dotted':
    for p in pts:
        cv2.circle(img, p, thickness, color, -1)
    # else:
    #     s = pts[0]
    #     e = pts[0]
    #     i = 0
    #     for p in pts:
    #         s = e
    #         e = p
    #         if i % 2 == 1:
    #             cv2.line(img, s, e, color, thickness)
    #         i += 1


def imshow(titles, frames, _pause):
    if isinstance(titles, str):
        titles = (titles,)
        frames = (frames,)

    for title, frame in zip(titles, frames):
        cv2.imshow(title, frame)

    if _pause > 1:
        wait = _pause
    else:
        wait = 1 - _pause

    k = cv2.waitKey(wait)

    if k == 27:
        sys.exit()
    elif k == 32:
        _pause = 1 - _pause

    return _pause, k


def draw_dotted_poly(img, pts, color, thickness=1):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        draw_dotted_line(img, s, e, color, thickness)


def draw_dotted_rect(img, pt1, pt2, color, thickness=1):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    draw_dotted_poly(img, pts, color, thickness)


def drawBox(image, xmin, ymin, xmax, ymax,
            box_color=(0, 255, 0), label=None, thickness=2, is_dotted=False):
    # if cv2.__version__.startswith('3'):
    #     font_line_type = cv2.LINE_AA
    # else:
    #     font_line_type = cv2.CV_AA

    if is_dotted:
        pt1 = (int(xmin), int(ymin))
        pt2 = (int(xmax), int(ymax))
        draw_dotted_rect(image, pt1, pt2, box_color, thickness=thickness)
    else:
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                      box_color, thickness=thickness)

    _bb = [xmin, ymin, xmax, ymax]
    if _bb[1] > 10:
        y_loc = int(_bb[1] - 5)
    else:
        y_loc = int(_bb[3] + 5)
    if label is not None:
        cv2.putText(image, label, (int(_bb[0] - 1), y_loc), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, box_color, 1, cv2.LINE_AA)


def writeCorners(file_id, corners, frame_id=-1, write_header=0):
    if write_header:
        file_id.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
    corner_str = ''
    for i in range(4):
        corner_str = corner_str + '{:5.2f}\t{:5.2f}\t'.format(corners[0, i], corners[1, i])
    if frame_id > 0:
        file_id.write('frame{:05d}.jpg\t'.format(frame_id))
    file_id.write(corner_str + '\n')


def writeCornersMOT(file_id, data, frame_id=None):
    if frame_id is None:
        frame_id = int(data[0])
    corner_str = '{:d},{:d}'.format(frame_id, int(data[1]))
    for i in range(2, 10):
        corner_str = corner_str + ',{:5.2f}'.format(data[i])
    file_id.write(corner_str + '\n')


def writeCorners2(file_name, corners, frame_id=-1, write_header=0):
    if write_header:
        file_id = open(file_name, 'w')
        file_id.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
    else:
        file_id = open(file_name, 'a')
    corner_str = ''
    for i in range(4):
        corner_str = corner_str + '{:5.2f}\t{:5.2f}\t'.format(corners[0, i], corners[1, i])
    if frame_id > 0:
        file_id.write('frame{:05d}.jpg\t'.format(frame_id))
    file_id.write(corner_str + '\n')
    file_id.close()


def getError(actual_corners, tracked_corners):
    curr_error = math.sqrt(np.sum(np.square(actual_corners - tracked_corners)) / 4)
    return curr_error

    # print 'inbuilt error: ', self.curr_error
    # self.curr_error=0
    # for i in range(actual_corners.shape[0]):
    # for j in range(actual_corners.shape[1]):
    # self.curr_error += math.pow(actual_corners[i, j] - tracked_corners[i, j], 2)
    # self.curr_error = math.sqrt(self.curr_error / 4)
    # print 'explicit error: ', self.curr_error


def getGroundTruthUpdates(filename):
    ground_truth = readTrackingData(filename)
    no_of_frames = ground_truth.shape[0]
    unit_square = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]).T
    last_corners = unit_square
    current_corners = None
    update_array = None
    updates = []
    update_filename = 'updates.txt'
    if os.path.exists(update_filename):
        os.remove(update_filename)
    update_file = open(update_filename, 'a')
    for i in range(no_of_frames):
        if current_corners is not None:
            last_corners = current_corners.copy()
        current_corners = np.array([ground_truth[i - 1, 0:2].tolist(),
                                    ground_truth[i - 1, 2:4].tolist(),
                                    ground_truth[i - 1, 4:6].tolist(),
                                    ground_truth[i - 1, 6:8].tolist()]).T
        update = compute_homography(last_corners, current_corners)
        # apply_to_pts(update, unit_square)
        # update.tofile(update_file)

        update = update.reshape((1, -1))
        update = np.delete(update, [8])
        # print 'update:\n', update
        np.savetxt(update_file, update, fmt='%12.8f', delimiter='\t')
        if update_array is None:
            if i > 0:
                update_array = np.asarray(update)
        else:
            update_array = np.append(update_array, update, axis=0)
        updates.append(update)
        # update_file.write('\n')
    # print 'updates:\n', updates
    update_file.close()
    # print 'update_array:\n', update_array
    np.savetxt('update_array.txt', update_array, fmt='%12.8f', delimiter='\t')
    # plotPCA(update_array)
    plotSuccessiveEuclideanDistance(update_array)
    return updates


def plotPCA(data):
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        print('PCA import failed: {}'.format(e))
        return
    # construct your numpy array of data
    pca = PCA(n_components=2)
    pca.fit(X)
    result = pca.components_
    # result = PCA(np.array(data))
    x = []
    y = []
    z = []
    for item in result.Y:
        x.append(item[0])
        y.append(item[1])
        z.append(item[2])

    plt.close('all')  # close all latent plotting windows
    fig1 = plt.figure()  # Make a plotting figure
    ax = Axes3D(fig1)  # use the plotting figure to create a Axis3D object.
    pltData = [x, y, z]
    ax.scatter(pltData[0], pltData[1], pltData[2], 'ko')  # make a scatter plot of blue dots from the data
    # ax.plot_wireframe(pltData[0], pltData[1], pltData[2]) # make a scatter plot of blue dots from the data
    # make simple, bare axis lines through space:
    xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0),
                 (0, 0))  # 2 points make the x-axis line at the data extrema along x-axis
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')  # make a red line for the x-axis.
    yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])),
                 (0, 0))  # 2 points make the y-axis line at the data extrema along y-axis
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'g')  # make a green line for the y-axis.
    zAxisLine = ((0, 0), (0, 0),
                 (min(pltData[2]), max(pltData[2])))  # 2 points make the z-axis line at the data extrema along z-axis
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'b')  # make a blue line for the z-axis.

    # label the axes
    ax.set_xlabel("x-axis label")
    ax.set_ylabel("y-axis label")
    ax.set_zlabel("y-axis label")
    ax.set_title("The title of the plot")
    plt.show()  # show the plot


def plotSuccessiveEuclideanDistance(data):
    print('data.shape=', data.shape)
    no_of_items = data.shape[0]
    data_dim = data.shape[1]

    # data1 = data[1:no_of_items - 1, :]
    # data2 = data[2:no_of_items, :]
    # euc_dist = np.sqrt(np.square(data1 - data2))

    x = range(no_of_items - 1)
    y = np.empty((no_of_items - 1, 1))

    for i in range(no_of_items - 1):
        y[i] = math.sqrt(np.sum(np.square(data[i + 1, :] - data[i, :])) / data_dim)

    plt.close('all')
    plt.figure()
    plt.plot(x, y)
    plt.show()


def getTrackingError(ground_truth_path, result_path, dataset, tracker_id):
    ground_truth_filename = ground_truth_path + '/' + dataset + '.txt'
    ground_truth_data = readTrackingData(ground_truth_filename)
    result_filename = result_path + '/' + dataset + '_res_%s.txt' % tracker_id

    result_data = readTrackingData(result_filename)
    [no_of_frames, no_of_pts] = ground_truth_data.shape
    error = np.zeros([no_of_frames, 1])
    # print "no_of_frames=", no_of_frames
    # print "no_of_pts=", no_of_pts
    if result_data.shape[0] != no_of_frames or result_data.shape[1] != no_of_pts:
        # print "no_of_frames 2=", result_data.shape[0]
        # print "no_of_pts 2=", result_data.shape[1]
        raise SyntaxError("Mismatch between ground truth and tracking result")

    error_filename = result_path + '/' + dataset + '_res_%s_error.txt' % tracker_id
    error_file = open(error_filename, 'w')
    for i in range(no_of_frames):
        data1 = ground_truth_data[i, :]
        data2 = result_data[i, :]
        for j in range(no_of_pts):
            error[i] += math.pow(data1[j] - data2[j], 2)
        error[i] = math.sqrt(error[i] / 4)
        error_file.write("%f\n" % error[i])
    error_file.close()
    return error


def extractColumn(filepath, filename, column, header_size=2):
    # print 'filepath=', filepath
    # print 'filename=', filename
    data_file = open(filepath + '/' + filename + '.txt', 'r')
    # remove header
    # read headersf
    for i in range(header_size):
        data_file.readline()
    lines = data_file.readlines()
    column_array = []
    line_id = 0
    for line in lines:
        # print(line)
        words = line.split()
        if (len(words) != 4):
            msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
            raise SyntaxError(msg)
        current_val = float(words[column])
        column_array.append(current_val)
        line_id += 1
    data_file.close()
    # print 'column_array=', column_array
    return column_array


def getThresholdRate(val_array, threshold, cmp_type):
    no_of_frames = len(val_array)
    if no_of_frames < 1:
        raise SystemExit('Error array is empty')
    thresh_count = 0
    for val in val_array:
        if cmp_type == 'less':
            if val <= threshold:
                thresh_count += 1
        elif cmp_type == 'more':
            if val >= threshold:
                thresh_count += 1

    rate = float(thresh_count) / float(no_of_frames) * 100
    return rate


def getThresholdVariations(res_dir, filename, val_type, show_plot=False,
                           min_thresh=0, diff=1, max_thresh=100, max_rate=100, agg_filename=None):
    print('Getting threshold variations for', val_type)
    if val_type == 'error':
        cmp_type = 'less'
        column = 2
    elif val_type == 'fps':
        cmp_type = 'more'
        column = 0
    else:
        raise SystemExit('Invalid value type')
    val_array = extractColumn(res_dir, filename, column)
    rates = []
    thresholds = []
    threshold = min_thresh
    const_count = 0
    rate = getThresholdRate(val_array, threshold, cmp_type)
    rates.append(rate)
    thresholds.append(threshold)
    while True:
        threshold += diff
        last_rate = rate
        rate = getThresholdRate(val_array, threshold, cmp_type)
        rates.append(rate)
        thresholds.append(threshold)
        if rate == last_rate:
            const_count += 1
        else:
            const_count = 0
        # print 'rate=', rate
        # if rate>=max_rate or const_count>=max_const or threshold>=max_thresh:
        # break
        if threshold >= max_thresh:
            break
    outfile = val_type + '_' + filename + '_' + str(min_thresh) + '_' + str(diff) + '_' + str(max_rate)
    data_array = np.array([thresholds, rates])
    full_name = res_dir + '/' + outfile + '.txt'
    np.savetxt(full_name, data_array.T, delimiter='\t', fmt='%11.5f')
    if agg_filename is not None:
        agg_filename = val_type + '_' + agg_filename
        agg_file = open('Results/' + agg_filename + '.txt', 'a')
        agg_file.write(full_name + '\n')
        agg_file.close()
    combined_fig = plt.figure()
    plt.plot(thresholds, rates, 'r-')
    plt.xlabel(threshold)
    plt.ylabel('Rate')
    # plt.title(plot_fname)
    plot_dir = res_dir + '/plots'
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    combined_fig.savefig(plot_dir + '/' + outfile, ext='png', bbox_inches='tight')

    if show_plot:
        plt.show()

    return rates, outfile


def aggregateDataFromFiles(list_filename, plot_filename, header_size=0):
    print('Aggregating data from ', list_filename, '...')
    line_styles = ['-', '--', '-.', ':', '+', '*', 'D', 'x', 's', 'p', 'o', 'v', '^']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # no_of_colors=32
    # colors=getRGBColors(no_of_colors)
    line_style_count = len(line_styles)
    color_count = len(colors)
    list_file = open('Results/' + list_filename + '.txt', 'r')
    header = list_file.readline()
    legend = header.split()
    filenames = list_file.readlines()
    list_file.close()
    # no_of_files=len(files)
    combined_fig = plt.figure()

    col_index = 0
    line_style_index = 0
    plot_lines = []
    for filename in filenames:
        filename = filename.rstrip()
        if not filename:
            continue
        # print 'filename=', filename
        data_file = open(filename, 'r')
        for i in range(header_size):
            data_file.readline()
        lines = data_file.readlines()
        data_file.close()
        os.remove(filename)
        # print 'data_str before =', data_str
        # data_str=np.asarray(data_str)
        # print 'data_str after =', data_str
        # if (len(data_str.shape) != 2):
        # print 'data_str.shape=', data_str.shape
        # raise SystemError('Error in aggregateDataFromFiles:\nInvalid syntax detected')
        thresholds = []
        rate_data = []
        for line in lines:
            words = line.split()
            threshold = float(words[0])
            rate = float(words[1])
            thresholds.append(threshold)
            rate_data.append(rate)
        # data_float=float(data_str)
        # thresholds=data_float[:, 0]
        # rate_data=data_float[:, 1]
        if col_index == color_count:
            col_index = 0
            line_style_index += 1
        if line_style_index == line_style_count:
            line_style_index = 0
        # plt.plot(thresholds, rate_data, color=colors[col_index], linestyle=line_styles[line_style_index])
        plt.plot(thresholds, rate_data, colors[col_index] + line_styles[line_style_index])
        col_index += 1
        # plot_lines.append(plot_line)
        # data_array=np.asarray([thresholds, rate_data])
        # combined_data.append()

    # plt.show()
    plt.xlabel('thresholds')
    plt.ylabel('rate')
    legend_dir = 'Results/legend'
    if not os.path.isdir(legend_dir):
        os.makedirs(legend_dir)
    # plt.title(plot_fname)
    fontP = ftm.FontProperties()
    fontP.set_size('small')
    combined_fig.savefig('Results/' + plot_filename, ext='png')
    plt.legend(legend, prop=fontP)
    combined_fig.savefig(legend_dir + '/' + plot_filename, ext='png')
    # plt.show()


def plotThresholdVariationsFromFile(filename, plot_fname):
    data_file = open('Results/' + filename, 'r')
    header = data_file.readline()
    header_words = header.split()
    lines = data_file.readlines()
    # print 'header_words=', header_words

    header_count = len(header_words)
    line_count = len(lines)

    data_array = np.empty((line_count, header_count))
    for i in range(line_count):
        # print(line)
        words = lines[i].split()
        if (len(words) != header_count):
            msg = "Invalid formatting on line %d" % i + " in file %s" % filename + ":\n%s" % lines[i]
            raise SyntaxError(msg)
        for j in range(header_count):
            data_array[i, j] = float(words[j])
    thresholds = data_array[:, 0]
    combined_fig = plt.figure(0)
    for i in range(1, header_count):
        rate_data = data_array[:, i]
        plt.plot(thresholds, rate_data)

    plt.xlabel(header_words[0])
    plt.ylabel('Success Rate')
    # plt.title(plot_fname)
    combined_fig.savefig('Results/' + plot_fname, ext='png', bbox_inches='tight')
    plt.legend(header_words[1:])
    combined_fig.savefig('Results/legend/' + plot_fname, ext='png', bbox_inches='tight')
    plt.show()


def getRGBColors(no_of_colors):
    channel_div = 0
    while no_of_colors > (channel_div ** 3):
        channel_div += 1
    colors = []
    if channel_div == 0:
        return colors
    base_factor = float(1.0 / float(channel_div))
    for i in range(channel_div):
        red = base_factor * i
        for j in range(channel_div):
            green = base_factor * j
            for k in range(channel_div):
                blue = base_factor * k
                col = (red, green, blue)
                colors.append(col)
    return colors


def readPerformanceSummary(filename, root_dir=None):
    # print 'filename=', filename
    if root_dir is None:
        root_dir = 'Results'
    data_file = open(root_dir + '/' + filename.rstrip() + '.txt', 'r')
    header = data_file.readline().split()
    success_rate_list = []
    avg_fps_list = []
    avg_drift_list = []
    parameters_list = []
    line_count = 0
    for line in data_file.readlines():
        line_count += 1
        words = line.split()
        success_rate = float(words[header.index('success_rate')])
        avg_fps = float(words[header.index('avg_fps')])
        avg_drift = float(words[header.index('avg_drift')])
        parameters = words[header.index('parameters')]

        success_rate_list.append(success_rate)
        avg_fps_list.append(avg_fps)
        avg_drift_list.append(avg_drift)
        parameters_list.append(parameters)
    data = {
        'parameters': parameters_list,
        'success_rate': success_rate_list,
        'avg_fps': avg_fps_list,
        'avg_drift': avg_drift_list
    }
    return header, data, line_count


def splitFiles(fname, keywords, root_dir=None, plot=False):
    if root_dir is None:
        root_dir = 'Results'
    list_file = open(root_dir + '/' + fname + '.txt')
    plot_fname = list_file.readline().rstrip()
    filenames = list_file.readlines()
    for keyword in keywords:
        new_list_filename = fname + '_' + keyword
        new_list_file = open(root_dir + '/' + new_list_filename + '.txt', 'w')
        new_list_file.write(plot_fname + '_' + keyword + '\n')
        for filename in filenames:
            filename = filename.rstrip()
            header, data, data_count = readPerformanceSummary(filename, root_dir)
            parameters = data['parameters']
            success_rate = data['success_rate']
            avg_fps = data['avg_fps']
            avg_drift = data['avg_drift']
            split_filename = filename + '_' + keyword
            split_file = open(root_dir + '/' + split_filename + '.txt', 'w')
            new_list_file.write(filename + '_' + keyword + '\n')
            for title in header:
                split_file.write(title + '\t')
            split_file.write('\n')
            for i in range(data_count):
                if keyword in parameters[i]:
                    split_file.write(parameters[i] + '\t')
                    split_file.write(str(success_rate[i]) + '\t')
                    split_file.write(str(avg_fps[i]) + '\t')
                    split_file.write(str(avg_drift[i]) + '\n')
            split_file.close()
        new_list_file.close()
        if plot:
            getPointPlot(root_dir=root_dir, file=new_list_filename, show_plot=True)


def getPointPlot(root_dir=None, filenames=None, plot_fname=None,
                 file=None, use_sep_fig=True, show_plot=False, legend=None, xticks=None,
                 title='', plot_drift=True):
    font = {'weight': 'bold',
            'size': 14}
    matplotlib.rc('font', **font)

    if root_dir is None:
        root_dir = 'Results'
    if show_plot:
        plt.close('all')
    if filenames is None:
        if file is None:
            return
        list_file = open(root_dir + '/' + file + '.txt')
        plot_fname = list_file.readline().rstrip()
        filenames = list_file.readlines()

    if title is None:
        title = plot_fname
    line_styles = ['-', '--', '-.', ':']
    markers = ['+', 'o', 'D', 'x', 's', 'p', '*', 'v', '^']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    fontP = ftm.FontProperties()
    fontP.set_size('small')

    if plot_drift:
        sub_plot_count = 3
    else:
        sub_plot_count = 2

    # ----------------------initializing success rate plot---------------------- #
    fig = plt.figure(0)
    fig.canvas.set_window_title('Success Rate')
    if not use_sep_fig:
        plt.subplot(sub_plot_count, 1, 1)
    plt.title(title + ' Success Rate')

    # ----------------------initializing fps plot---------------------- #
    # plt.legend(filenames)
    if use_sep_fig:
        fig = plt.figure(1)
        fig.canvas.set_window_title('Average FPS')
    else:
        plt.subplot(sub_plot_count, 1, 2)
    plt.title(title + ' Average FPS')

    # ----------------------initializing drift plot---------------------- #
    if plot_drift:
        # plt.legend(filenames)
        if use_sep_fig:
            fig = plt.figure(2)
            fig.canvas.set_window_title('Average Drift')
        else:
            plt.subplot(sub_plot_count, 1, 3)
        plt.title(title + ' Average Drift')
    # plt.legend(filenames)

    # annotate_text_list=None
    linestyle_id = 0
    marker_id = 0
    color_id = 0
    success_rate_y = range(0, 200, 5)
    avg_fps_y = range(0, 100, 5)
    print('success_rate_y=', success_rate_y)
    print('avg_fps_y=', avg_fps_y)

    for filename in filenames:
        print('filename=', filename)
        header, data, data_count = readPerformanceSummary(filename, root_dir=root_dir)
        # parameters=data['parameters']
        success_rate = data['success_rate']
        avg_fps = data['avg_fps']
        avg_drift = data['avg_drift']

        x = range(0, data_count)

        # ----------------------updating success rate plot---------------------- #
        if use_sep_fig:
            plt.figure(0, figsize=(1920 / 96, 1080 / 96), dpi=96)
        else:
            plt.subplot(sub_plot_count, 1, 1)
        if xticks is None:
            plt.xticks(x, map(str, x))
        else:
            plt.xticks(x, xticks)
        plt.yticks(success_rate_y)
        plt.plot(x, success_rate,
                 colors[color_id] + markers[marker_id] + line_styles[linestyle_id])

        # ----------------------updating fps plot---------------------- #
        if use_sep_fig:
            plt.figure(1)
        else:
            plt.subplot(sub_plot_count, 1, 2)
        if xticks is None:
            plt.xticks(x, map(str, x))
        else:
            plt.xticks(x, xticks)
        plt.yticks(avg_fps_y)
        plt.plot(x, avg_fps,
                 colors[color_id] + markers[marker_id] + line_styles[linestyle_id])

        # ----------------------updating drift plot---------------------- #
        if plot_drift:
            if use_sep_fig:
                plt.figure(2)
            else:
                plt.subplot(sub_plot_count, 1, 3)
            if xticks is None:
                plt.xticks(x, map(str, x))
            else:
                plt.xticks(x, xticks)
            plt.plot(x, avg_drift,
                     colors[color_id] + markers[marker_id] + line_styles[linestyle_id])

        color_id = (color_id + 1) % len(colors)
        marker_id = (marker_id + 1) % len(markers)
        linestyle_id = (linestyle_id + 1) % len(line_styles)
        # annotate_text_list=parameters
    # annotate_text=''
    # print 'annotate_text_list:\n', annotate_text_list
    # for i in range(len(annotate_text_list)):
    # annotate_text=annotate_text+str(i)+': '+annotate_text_list[i]+'\n'
    #
    # print 'annotate_text=\n', annotate_text

    # ----------------------saving success rate plot---------------------- #
    if use_sep_fig:
        plt.figure(0)
    else:
        plt.subplot(sub_plot_count, 1, 1)
    ax = plt.gca()
    # for tick in ax.xaxis.get_major_ticks():
    # tick.label.set_fontsize(12)
    # specify integer or one of preset strings, e.g.
    # tick.label.set_fontsize('x-small')
    # tick.label.set_rotation('vertical')
    if legend is None:
        plt.legend(filenames, prop=fontP)
    else:
        plt.legend(legend, prop=fontP)
    plt.grid(True)
    # plt.figtext(0.01,0.01, annotate_text, fontsize=9)
    if use_sep_fig and plot_fname is not None:
        plt.savefig(root_dir + '/' + plot_fname + '_success_rate', dpi=96, ext='png')

    # ----------------------saving fps plot---------------------- #
    if use_sep_fig:
        plt.figure(1)
    else:
        plt.subplot(sub_plot_count, 1, 2)

    if legend is None:
        plt.legend(filenames, prop=fontP)
    else:
        plt.legend(legend, prop=fontP)
    plt.grid(True)
    if use_sep_fig and plot_fname is not None:
        plt.savefig(root_dir + '/' + plot_fname + '_avg_fps', dpi=96, ext='png')

    # ----------------------saving drift plot---------------------- #
    if plot_drift:
        if use_sep_fig:
            plt.figure(2)
        else:
            plt.subplot(sub_plot_count, 1, 3)
        if legend is None:
            plt.legend(filenames, prop=fontP)
        else:
            plt.legend(legend, prop=fontP)
        plt.grid(True)
        if use_sep_fig and plot_fname is not None:
            plt.savefig(root_dir + '/' + plot_fname + '_avg_drft', dpi=96, ext='png')

    # ----------------------saving combined plot---------------------- #
    if not use_sep_fig and plot_fname is not None:
        plt.savefig(root_dir + '/' + plot_fname, dpi=96, ext='png')

    if show_plot:
        plt.show()


class InteractivePlot:
    def __init__(self, root_dir=None, filenames=None, plot_fname=None,
                 file=None, legend=None, title=None, xticks=None, y_tick_count=10):
        plt.close('all')

        font = {'weight': 'bold',
                'size': 14}
        matplotlib.rc('font', **font)

        if root_dir is None:
            root_dir = 'Results'

        if filenames is None:
            if file is None:
                return
            list_file = open(root_dir + '/' + file + '.txt')
            plot_fname = list_file.readline().rstrip()
            filenames = list_file.readlines()

        if legend is None:
            legend = filenames

        if title is None:
            title = plot_fname

        self.legend = legend
        self.title = title.replace('_', ' ').title()

        # plotting state variables
        self.plot_types = ['success_rate', 'avg_fps', 'avg_drift']
        self.plot_id = 0
        self.active_line = 0
        self.plot_all_lines = True
        self.exit_event = False
        self.no_of_lines = len(filenames)
        self.y_tick_count = y_tick_count
        self.show_lines = [1] * self.no_of_lines

        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 'D', '+', 'x', 's', 'p', '*', 'v', '^']
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        linestyle_id = 0
        marker_id = 0
        color_id = 0

        self.fontP = ftm.FontProperties()
        self.fontP.set_size('small')

        self.fig = plt.figure(0)
        # self.fig.canvas.set_window_title('Success Rate')
        cid = self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.ax = plt.axes()
        # plt.ylabel('Run')
        plt.grid(True)
        # plt.show()

        self.plot_lines = []
        self.plot_data = []
        self.line_patterns = []

        num_keys = [str(i) for i in range(0, 10)]
        ctrl_keys = ['ctrl+' + str(i) for i in range(0, 10)]
        alt_keys = ['alt+' + str(i) for i in range(0, 10)]
        ctrl_alt_keys = ['ctrl+alt+' + str(i) for i in range(0, 10)]
        # print 'ctrl_keys=', ctrl_keys
        self.ctrl_keys = dict(zip(num_keys + alt_keys + ctrl_keys + ctrl_alt_keys,
                                  range(0, 10) * 2 + range(10, 20) * 2))

        file_id = 0
        for filename in filenames:
            # print 'filename=', filename
            header, data, data_count = readPerformanceSummary(filename, root_dir=root_dir)
            self.legend[file_id] = str(file_id) + ': ' + self.legend[file_id].rstrip()
            # parameters=data['parameters']
            self.plot_data.append(data)

            # ----------------------initializing plot---------------------- #
            self.x_data = range(0, data_count)
            y_data = data[self.plot_types[self.plot_id]]
            # print 'y_data=', y_data
            line_pattern = colors[color_id] + markers[marker_id] + line_styles[linestyle_id]
            self.line_patterns.append(line_pattern)
            line, = self.ax.plot(self.x_data, y_data, line_pattern, label=self.legend[file_id])
            self.plot_lines.append(line)

            color_id += 1
            if color_id >= len(colors):
                color_id = 0
                linestyle_id = (linestyle_id + 1) % len(line_styles)
                marker_id = (marker_id + 1) % len(markers)
            file_id += 1

        if xticks is None:
            plt.xticks(self.x_data, map(str, self.x_data))
        else:
            plt.xticks(self.x_data, xticks)

        # self.ax.legend(self.legend, prop=self.fontP)
        self.ax.legend(prop=self.fontP)
        anim = animation.FuncAnimation(self.fig, self.animate, self.simData, init_func=self.initPlot)

        # fig_drift.canvas.draw()
        plt.show()

    def onKeyPress(self, event):
        print('key pressed=', event.key)
        if event.key == "escape" or event.key == "alt+escape":
            self.exit_event = True
            sys.exit()
        elif event.key == "down" or event.key == "alt+down":
            if self.plot_all_lines:
                self.plot_all_lines = False
                return
            self.active_line = (self.active_line + 1) % self.no_of_lines
        elif event.key == "up" or event.key == "alt+up":
            if self.plot_all_lines:
                self.plot_all_lines = False
                return
            self.active_line -= 1
            if self.active_line < 0:
                self.active_line = self.no_of_lines - 1
        elif event.key == "shift" or event.key == "alt+shift":
            self.plot_all_lines = not self.plot_all_lines
        elif event.key == "right" or event.key == "alt+right":
            self.plot_id = (self.plot_id + 1) % len(self.plot_types)
        elif event.key == "left" or event.key == "alt+left":
            self.plot_id -= 1
            if self.plot_id < 0:
                self.plot_id = len(self.plot_types) - 1
        elif event.key in self.ctrl_keys.keys():
            if not self.plot_all_lines:
                return
            key_id = self.ctrl_keys[event.key]
            if key_id < self.no_of_lines:
                self.show_lines[key_id] = 1 - self.show_lines[key_id]
                if not self.show_lines[key_id]:
                    print('Removing line for', self.legend[key_id])
                else:
                    print('Restoring line for', self.legend[key_id])
        elif event.key == "i" or event.key == "alt+i":
            self.show_lines = [1 - x for x in self.show_lines]
        elif event.key == "r" or event.key == "alt+r":
            self.show_lines = [1] * self.no_of_lines

    def simData(self):
        yield 1

    def initPlot(self):
        return self.plot_lines

    def animate(self, i):
        if self.exit_event:
            sys.exit()
        plot_title = self.plot_types[self.plot_id].replace('_', ' ').title()

        plot_empty = True
        max_y = 0
        if self.plot_all_lines:
            for i in range(self.no_of_lines):
                y_data = self.plot_data[i][self.plot_types[self.plot_id]]
                curr_y_max = max(y_data)
                if max_y < curr_y_max:
                    max_y = curr_y_max
                # print 'y_data=', y_data
                if self.show_lines[i]:
                    plot_empty = False
                    self.plot_lines[i].set_data(self.x_data, y_data)
                    self.plot_lines[i].set_label(self.legend[i])
                else:
                    self.plot_lines[i].set_data([], [])
                    self.plot_lines[i].set_label('_' + self.legend[i])
                    # self.ax.legend((lines, legend), prop=self.fontP)
                    # self.ax.get_legend().set_visible(True)
        else:
            plot_empty = False
            for i in range(self.no_of_lines):
                self.plot_lines[i].set_data([], [])
                self.plot_lines[i].set_label('_' + self.legend[i])
            y_data = self.plot_data[self.active_line][self.plot_types[self.plot_id]]
            # print 'y_data=', y_data
            self.plot_lines[self.active_line].set_data(self.x_data, y_data)
            self.plot_lines[self.active_line].set_label(self.legend[self.active_line])
            max_y = max(y_data)
            # plot_title = plot_title + '_' + self.legend[self.active_line]
            # self.ax.get_legend().set_visible(False)

        if plot_empty:
            self.ax.get_legend().set_visible(False)
        else:
            self.ax.legend(prop=self.fontP)
        plot_title = plot_title + ' for ' + self.title
        self.fig.canvas.set_window_title(plot_title)
        plt.title(plot_title)
        self.ax.set_ylim(0, max_y)
        y_diff = int(math.ceil(max_y / self.y_tick_count))
        y_ticks = range(0, y_diff * self.y_tick_count + 1, y_diff)
        # print 'max_y=', max_y
        # print 'y_diff=', y_diff
        # print 'y_ticks=', y_ticks
        plt.yticks(y_ticks, map(str, y_ticks))

        plt.draw()
        return self.plot_lines


def lineIntersection(line1, line2):
    r1 = line1[0]
    theta1 = line1[1]
    r2 = line2[0]
    theta2 = line2[1]

    if theta1 == theta2:
        raise StandardError('Lines are parallel')
    elif theta1 == 0:
        x = r1
        y = -x * (math.cos(theta2) / math.sin(theta2)) + (r2 / math.sin(theta2))
    elif theta2 == 0:
        x = r2
        y = -x * (math.cos(theta1) / math.sin(theta1)) + (r1 / math.sin(theta1))
    else:
        sin_theta1 = math.sin(theta1)
        cos_theta1 = math.cos(theta1)
        sin_theta2 = math.sin(theta2)
        cos_theta2 = math.cos(theta2)

        m1 = -cos_theta1 / sin_theta1
        c1 = r1 / sin_theta1
        m2 = -cos_theta2 / sin_theta2
        c2 = r2 / sin_theta2

        x = (c1 - c2) / (m2 - m1)
        y = m1 * x + c1
        # print 'r1: ', r1, 'theta1: ', theta1
        # print 'r2: ', r2, 'theta2: ', theta2
        # print 'sin_theta1: ', sin_theta1, 'cos_theta1: ', cos_theta1
        # print 'sin_theta2: ', sin_theta2, 'cos_theta2: ', cos_theta2
        # print 'm1: ', m1, 'c1: ', c1
        # print 'm2: ', m2, 'c2: ', c2
        # print 'x: ', x, 'y: ', y
        # print '\n'
    return x, y


def getIntersectionPoints(lines_arr):
    no_of_lines = len(lines_arr)
    if no_of_lines != 4:
        raise StandardError('Invalid number of lines provided: ' + str(no_of_lines))
    # line1 = lines_arr[0, :]
    # theta_diff = np.fabs(line1[1] - lines_arr[:, 1])

    pi = cv2.cv.CV_PI
    pi_2 = cv2.cv.CV_PI / 2.0

    min_theta_diff = np.inf
    min_i = 0
    min_j = 0
    for i in range(4):
        theta1 = lines_arr[i, 1]
        # if theta1 > pi_2 and lines_arr[i, 0] < 0:
        # theta1 = pi - theta1
        for j in range(i + 1, 4):
            theta2 = lines_arr[j, 1]
            # if theta2 > pi_2 and lines_arr[j, 0] < 0:
            # theta2 = pi - theta2
            theta_diff = math.fabs(theta1 - theta2)
            if theta_diff < min_theta_diff:
                min_theta_diff = theta_diff
                min_i = i
                min_j = j
    line1 = lines_arr[min_i, :]
    line2 = lines_arr[min_j, :]

    print('before: lines_arr:\n', lines_arr)
    print('min_i: ', min_i)
    print('min_j: ', min_j)
    print('line1: ', line1)
    print('line2: ', line2)

    pts = []
    for i in range(4):
        if i != min_i and i != min_j:
            print('getting intersection between lines {:d} and {:d}'.format(min_i, i))
            pt = lineIntersection(line1, lines_arr[i, :])
            print('intersection pt: ', pt)
            pts.append(pt)
            print('getting intersection between lines {:d} and {:d}'.format(min_j, i))
            pt = lineIntersection(line2, lines_arr[i, :])
            print('intersection pt: ', pt)
            pts.append(pt)
    pt_arr = np.array(pts)

    # print 'theta_diff: \n', theta_diff

    # sort params by theta_diff
    # lines_arr = lines_arr[theta_diff.argsort()]
    # print 'after: lines_arr:\n', lines_arr
    # pt1 = lineIntersection(lines_arr[0, :], lines_arr[2, :])
    # pt2 = lineIntersection(lines_arr[0, :], lines_arr[3, :])
    # pt3 = lineIntersection(lines_arr[1, :], lines_arr[2, :])
    # pt4 = lineIntersection(lines_arr[1, :], lines_arr[3, :])

    # pt_arr = np.array([pt1, pt2, pt3, pt4])
    # print 'pt_arr:\n', pt_arr
    pt_arr_sorted = pt_arr[pt_arr[:, 0].argsort()]
    # print 'pt_arr_sorted:\n', pt_arr_sorted

    if pt_arr_sorted[0, 1] < pt_arr_sorted[1, 1]:
        ulx, uly = pt_arr_sorted[0, :]
        urx, ury = pt_arr_sorted[1, :]
    else:
        ulx, uly = pt_arr_sorted[1, :]
        urx, ury = pt_arr_sorted[0, :]

    if pt_arr_sorted[2, 1] < pt_arr_sorted[3, 1]:
        llx, lly = pt_arr_sorted[2, :]
        lrx, lry = pt_arr_sorted[3, :]
    else:
        llx, lly = pt_arr_sorted[3, :]
        lrx, lry = pt_arr_sorted[2, :]

    return ulx, uly, urx, ury, lrx, lry, llx, lly


def refineCorners(corners, curr_img_gs):
    ulx, uly = corners[:, 0]
    urx, ury = corners[:, 1]
    lrx, lry = corners[:, 2]
    llx, lly = corners[:, 3]

    return corners


def readMetaioInitData(init_file):
    gt_corners = []

    init_fid = open(init_file, 'r')
    lines = init_fid.readlines()
    init_fid.close()

    curr_corners = [0] * 8
    gt_id = 0
    for line in lines:
        words = line.split()
        if words is None or len(words) <= 1:
            continue
        if isNumber(words[0]):
            curr_corners[gt_id] = float(words[0])
            curr_corners[gt_id + 1] = float(words[1])
            gt_id += 2
        elif gt_id > 0:
            gt_corners.append(copy.deepcopy(curr_corners))
            gt_id = 0

    gt_corners.append(copy.deepcopy(curr_corners))
    gt_corners_array = np.array(gt_corners, dtype=np.float64)
    # print 'gt_corners: ', gt_corners
    # print 'gt_corners_array: ', gt_corners_array
    return gt_corners_array


def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def getLinearInterpolatedImages(init_img, final_img, count):
    interpolated_images = []
    for i in range(count):
        alpha = float(i + 1) / float(count + 1)
        curr_img = alpha * final_img + (1.0 - alpha) * init_img
        interpolated_images.append(np.copy(curr_img.astype(np.float64)))

    return interpolated_images


def readDistGridParams(filename='distanceGridParams.txt'):
    dicts_from_file = {}
    with open(filename, 'r') as param_file:
        for line in param_file:
            words = line.split()
            dicts_from_file[words[0]] = int(words[1])
    return dicts_from_file


def readGT(gt_path):
    gt_data = open(gt_path, 'r').readlines()
    if len(gt_data) < 2:
        print('Ground truth file is invalid')
        return None, None
    del (gt_data[0])
    n_lines = len(gt_data)
    gt = []
    line_id = 0
    while line_id < n_lines:
        gt_line = gt_data[line_id].strip().split()
        gt_frame_fname = gt_line[0]
        gt_frame_num = int(gt_frame_fname[5:-4])
        gt_frame_fname_1 = gt_frame_fname[0:5]
        gt_frame_fname_2 = gt_frame_fname[- 4:]
        if len(gt_line) != 9 \
                or gt_frame_fname_1 != 'frame' \
                or gt_frame_fname_2 != '.jpg' \
                or gt_frame_num != line_id + 1:
            print('Invaid formatting on GT  line {:d}: {:s}'.format(line_id + 1, gt_line))
            print('gt_frame_fname_1: {:s}'.format(gt_frame_fname_1))
            print('gt_frame_fname_2: {:s}'.format(gt_frame_fname_2))
            print('gt_frame_num: {:d}'.format(gt_frame_num))
            return None, None
        gt.append([float(gt_line[1]), float(gt_line[2]), float(gt_line[3]), float(gt_line[4]),
                   float(gt_line[5]), float(gt_line[6]), float(gt_line[7]), float(gt_line[8])])
        line_id += 1

    return n_lines, gt


def readReinitGT(gt_path, reinit_frame_id):
    gt_fid = open(gt_path, 'rb')
    try:
        n_gt_frames = struct.unpack('i', gt_fid.read(4))[0]
    except struct.error:
        gt_fid.close()
        raise StandardError("Reinit GT file is invalid")
    print('Reading reinit gt for frame {:d}'.format(reinit_frame_id + 1))
    start_pos = reinit_frame_id * (2 * n_gt_frames - reinit_frame_id + 1) * 4 * 8 + 4
    gt_fid.seek(start_pos)
    reinit_gt = []
    for frame_id in range(reinit_frame_id, n_gt_frames):
        try:
            curr_gt = struct.unpack('dddddddd', gt_fid.read(64))
        except struct.error:
            gt_fid.close()
            raise StandardError("Reinit GT file is invalid")
        reinit_gt.append([
            curr_gt[0], curr_gt[4],
            curr_gt[1], curr_gt[5],
            curr_gt[2], curr_gt[6],
            curr_gt[3], curr_gt[7]
        ])
    gt_fid.close()
    return n_gt_frames, reinit_gt


def getMeanCornerDistanceError(tracker_pos, gt_pos, _overflow_err=1e3):
    # mean corner distance error
    err = 0
    for corner_id in range(4):
        try:
            # err += math.sqrt(
            # (float(tracking_data_line[2 * corner_id + 1]) - float(gt_line[2 * corner_id + 1])) ** 2
            # + (float(tracking_data_line[2 * corner_id + 2]) - float(gt_line[2 * corner_id + 2])) ** 2
            # )
            err += math.sqrt(
                (tracker_pos[2 * corner_id] - gt_pos[2 * corner_id]) ** 2
                + (tracker_pos[2 * corner_id + 1] - gt_pos[2 * corner_id + 1]) ** 2
            )
        except OverflowError:
            err += _overflow_err
            continue
    err /= 4.0
    # for corner_id in range(1, 9):
    # try:
    # err += (float(tracking_data_line[corner_id]) - float(gt_line[corner_id])) ** 2
    # except OverflowError:
    # continue
    # err = math.sqrt(err / 4)
    return err


def getCenterLocationError(tracker_pos, gt_pos):
    # center location error
    # centroid_tracker_x = (float(tracking_data_line[1]) + float(tracking_data_line[3]) + float(
    # tracking_data_line[5]) + float(tracking_data_line[7])) / 4.0
    # centroid2_x = (float(gt_line[1]) + float(gt_line[3]) + float(gt_line[5]) + float(
    # gt_line[7])) / 4.0

    centroid_tracker_x = (tracker_pos[0] + tracker_pos[2] + tracker_pos[4] +
                          tracker_pos[6]) / 4.0
    centroid_gt_x = (gt_pos[0] + gt_pos[2] + gt_pos[4] + gt_pos[6]) / 4.0

    # centroid_tracker_y = (float(tracking_data_line[2]) + float(tracking_data_line[4]) + float(
    # tracking_data_line[6]) + float(tracking_data_line[8])) / 4.0
    # centroid2_y = (float(gt_line[2]) + float(gt_line[4]) + float(gt_line[6]) + float(
    # gt_line[8])) / 4.0

    centroid_tracker_y = (tracker_pos[1] + tracker_pos[3] + tracker_pos[5] +
                          tracker_pos[7]) / 4.0
    centroid_gt_y = (gt_pos[1] + gt_pos[3] + gt_pos[5] + gt_pos[7]) / 4.0

    err = math.sqrt((centroid_tracker_x - centroid_gt_x) ** 2 + (centroid_tracker_y - centroid_gt_y) ** 2)
    # print 'tracking_data_line: ', tracking_data_line
    # print 'gt_line: ', gt_line
    # print 'centroid1_x: {:15.9f} centroid1_y:  {:15.9f}'.format(centroid1_x, centroid1_y)
    # print 'centroid2_x: {:15.9f} centroid2_y:  {:15.9f}'.format(centroid2_x, centroid2_y)
    # print 'err: {:15.9f}'.format(err)

    return err


def getJaccardError(tracker_pos, gt_pos, show_img=0, border_size=100, min_thresh=0, max_thresh=2000):
    min_x = int(min([tracker_pos[0], tracker_pos[2], tracker_pos[4], tracker_pos[6],
                     gt_pos[0], gt_pos[2], gt_pos[4], gt_pos[6]]))
    min_y = int(min([tracker_pos[1], tracker_pos[3], tracker_pos[5], tracker_pos[7],
                     gt_pos[1], gt_pos[3], gt_pos[5], gt_pos[7]]))
    max_x = int(max([tracker_pos[0], tracker_pos[2], tracker_pos[4], tracker_pos[6],
                     gt_pos[0], gt_pos[2], gt_pos[4], gt_pos[6]]))
    max_y = int(max([tracker_pos[1], tracker_pos[3], tracker_pos[5], tracker_pos[7],
                     gt_pos[1], gt_pos[3], gt_pos[5], gt_pos[7]]))

    if min_x < min_thresh:
        min_x = min_thresh
    if min_y < min_thresh:
        min_y = min_thresh
    if max_x > max_thresh:
        max_x = max_thresh
    if max_y > max_thresh:
        max_y = max_thresh

    if min_x > max_x or min_y > max_y:
        print('tracker_pos: ', tracker_pos)
        print('gt_pos: ', gt_pos)
        raise SystemError('Invalid Tracker and/or GT position')

    img_size = (max_y - min_y + 2 * border_size + 1, max_x - min_x + 2 * border_size + 1)

    tracker_pos_pts = np.asarray(
        [[tracker_pos[0] + border_size - min_x, tracker_pos[2] + border_size - min_x,
          tracker_pos[4] + border_size - min_x, tracker_pos[6] + border_size - min_x],
         [tracker_pos[1] + border_size - min_y, tracker_pos[3] + border_size - min_y,
          tracker_pos[5] + border_size - min_y, tracker_pos[7] + border_size - min_y]]
    )
    gt_pos_pts = np.asarray(
        [[gt_pos[0] + border_size - min_x, gt_pos[2] + border_size - min_x,
          gt_pos[4] + border_size - min_x, gt_pos[6] + border_size - min_x],
         [gt_pos[1] + border_size - min_y, gt_pos[3] + border_size - min_y,
          gt_pos[5] + border_size - min_y, gt_pos[7] + border_size - min_y]]
    )

    tracker_img = getBinaryPtsImage2(img_size, tracker_pos_pts)
    gt_img = getBinaryPtsImage2(img_size, gt_pos_pts)

    intersection_img = cv2.bitwise_and(tracker_img, gt_img)
    union_img = cv2.bitwise_or(tracker_img, gt_img)
    n_intersectio_pix = np.sum(intersection_img)
    n_union_pix = np.sum(union_img)
    jacc_error = 1.0 - float(n_intersectio_pix) / float(n_union_pix)

    if show_img:
        legend_font_size = 1
        legend_font_thickness = 1
        legend_font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
        if cv2.__version__.startswith('2'):
            legend_font_line_type = cv2.CV_AA
        else:
            legend_font_line_type = cv2.LINE_AA
        header_location = (0, 20)

        cv2.putText(tracker_img, '{:f}'.format(jacc_error), header_location, legend_font_face,
                    legend_font_size, col_rgb['white'], legend_font_thickness, legend_font_line_type)
        cv2.putText(intersection_img, '{:d}'.format(n_intersectio_pix), header_location, legend_font_face,
                    legend_font_size, col_rgb['white'], legend_font_thickness, legend_font_line_type)
        cv2.putText(union_img, '{:d}'.format(n_union_pix), header_location, legend_font_face,
                    legend_font_size, col_rgb['white'], legend_font_thickness, legend_font_line_type)
        cv2.imshow('tracker_img', tracker_img)
        cv2.imshow('gt_img', gt_img)
        cv2.imshow('intersection_img', intersection_img)
        cv2.imshow('union_img', union_img)

        if cv2.waitKey(1) == 27:
            sys.exit(0)
    return jacc_error


def getTrackingErrors(tracker_path_orig, gt_path, _arch_fid=None, _reinit_from_gt=0,
                      _reinit_frame_skip=5, _use_reinit_gt=0, start_ids=None, _err_type=0,
                      _overflow_err=1e3, _show_jaccard_img=0):
    print('Reading ground truth from: {:s}...'.format(gt_path))
    if _use_reinit_gt:
        n_gt_frames, gt_data = readReinitGT(gt_path, 0)
    else:
        n_gt_frames, gt_data = readGT(gt_path)

    if n_gt_frames is None or gt_data is None:
        print("Ground truth could not be read successfully")
        return None, None

    if start_ids is None:
        start_ids = [0]

    tracking_errors = []
    failure_count = 0
    for start_id in start_ids:
        if start_id == 0:
            tracker_path = tracker_path_orig
        else:
            tracker_path = tracker_path_orig.replace('.txt', '_init_{:d}.txt'.format(start_id))
        print('Reading tracking data for start_id {:d} from: {:s}...'.format(start_id, tracker_path))
        if _arch_fid is not None:
            tracking_data = _arch_fid.open(tracker_path, 'r').readlines()
        else:
            tracking_data = open(tracker_path, 'r').readlines()
        if len(tracking_data) < 2:
            print('Tracking data file is invalid.')
            return None, None
        # remove header
        del (tracking_data[0])
        n_lines = len(tracking_data)

        if not _reinit_from_gt and n_lines != n_gt_frames - start_id:
            print("No. of frames in tracking result ({:d}) and the ground truth ({:d}) do not match".format(
                n_lines, n_gt_frames))
            return None, None

        reinit_gt_id = 0
        reinit_start_id = 0
        # ignore the first frame where tracker was initialized
        line_id = 1
        invalid_tracker_state_found = False
        is_initialized = True
        # id of the last frame where tracking failure was detected
        failure_frame_id = -1

        while line_id < n_lines:
            tracking_data_line = tracking_data[line_id].strip().split()
            frame_fname = str(tracking_data_line[0])
            fname_len = len(frame_fname)
            frame_fname_1 = frame_fname[0:5]
            frame_fname_2 = frame_fname[- 4:]
            if frame_fname_1 != 'frame' or frame_fname_2 != '.jpg':
                print('Invaid formatting on tracking data line {:d}: {:s}'.format(line_id + 1, tracking_data_line))
                print('frame_fname: {:s} fname_len: {:d} frame_fname_1: {:s} frame_fname_2: {:s}'.format(
                    frame_fname, fname_len, frame_fname_1, frame_fname_2))
                return None, None
            frame_id_str = frame_fname[5:-4]
            frame_num = int(frame_id_str)
            if not _reinit_from_gt and frame_num != start_id + line_id + 1:
                print(
                    "Unexpected frame number {:d} found in line {:d} of tracking result for start_id {:d}: {:s}".format(
                        frame_num, line_id + 1, start_id, tracking_data_line))
                return None, None
            if is_initialized:
                # id of the frame in which the tracker is reinitialized
                reinit_start_id = frame_num - 2
                if failure_frame_id >= 0 and reinit_start_id != failure_frame_id + _reinit_frame_skip:
                    print(
                        'Tracker was reinitialized in frame {:d} rather than {:d} where it should have been with {:d} '
                        'frames being skipped'.format(
                            reinit_start_id + 1, failure_frame_id + _reinit_frame_skip + 1, _reinit_frame_skip
                        ))
                    return None, None
                is_initialized = False

            # print('line_id: {:d} frame_id_str: {:s} frame_num: {:d}'.format(
            # line_id, frame_id_str, frame_num)
            if len(tracking_data_line) != 9:
                if _reinit_from_gt and len(tracking_data_line) == 2 and tracking_data_line[1] == 'tracker_failed':
                    print('tracking failure detected in frame: {:d} at line {:d}'.format(frame_num, line_id + 1))
                    failure_count += 1
                    failure_frame_id = frame_num - 1
                    # skip the frame where the tracker failed as well as the one where it was reinitialized
                    # whose result will (or should) be in the line following this one
                    line_id += 2
                    is_initialized = True
                    continue
                elif len(tracking_data_line) == 2 and tracking_data_line[1] == 'invalid_tracker_state':
                    if not invalid_tracker_state_found:
                        print('invalid tracker state detected in frame: {:d} at line {:d}'.format(
                            frame_num, line_id + 1))
                        invalid_tracker_state_found = True
                    line_id += 1
                    tracking_errors.append(_overflow_err)
                    continue
                else:
                    print('Invalid formatting on line {:d}: {:s}'.format(line_id, tracking_data[line_id]))
                    return None, None
            # if is_initialized:frame_num
            # is_initialized = False
            # line_id += 1
            # continue

            if _use_reinit_gt:
                if reinit_gt_id != reinit_start_id:
                    n_gt_frames, gt_data = readReinitGT(gt_path, reinit_start_id)
                    reinit_gt_id = reinit_start_id
                curr_gt = gt_data[frame_num - reinit_start_id - 1]
            else:
                curr_gt = gt_data[frame_num - 1]

            curr_tracking_data = [float(tracking_data_line[1]), float(tracking_data_line[2]),
                                  float(tracking_data_line[3]), float(tracking_data_line[4]),
                                  float(tracking_data_line[5]), float(tracking_data_line[6]),
                                  float(tracking_data_line[7]), float(tracking_data_line[8])]
            # print 'line_id: {:d} gt: {:s}'.format(line_id, gt_line)

            if _err_type == 0:
                err = getMeanCornerDistanceError(curr_tracking_data, curr_gt, _overflow_err)
            elif _err_type == 1:
                err = getCenterLocationError(curr_tracking_data, curr_gt)
            elif _err_type == 2:
                err = getJaccardError(curr_tracking_data, curr_gt, _show_jaccard_img)
            else:
                print('Invalid error type provided: {:d}'.format(_err_type))
                return None, None
            tracking_errors.append(err)
            line_id += 1
        if _reinit_from_gt and n_lines < n_gt_frames - failure_count * (_reinit_frame_skip - 1):
            print("Unexpected no. of frames in reinit tracking result ({:d}) which should be at least {:d}".format(
                n_lines, n_gt_frames - failure_count * (_reinit_frame_skip - 1)))
            return None, None
    return tracking_errors, failure_count


def arrangeCorners(orig_corners):
    # print 'orig_corners:\n', orig_corners
    # print 'orig_corners.shape:\n', orig_corners.shape

    new_corners = np.zeros((2, 4), dtype=np.float64)
    sorted_x_id = np.argsort(orig_corners[0, :])

    # print 'new_corners.shape:\n', new_corners.shape
    # print 'sorted_x_id:\n', sorted_x_id

    if orig_corners[1, sorted_x_id[0]] < orig_corners[1, sorted_x_id[1]]:
        new_corners[:, 0] = orig_corners[:, sorted_x_id[0]]
        new_corners[:, 3] = orig_corners[:, sorted_x_id[1]]
    else:
        new_corners[:, 0] = orig_corners[:, sorted_x_id[1]]
        new_corners[:, 3] = orig_corners[:, sorted_x_id[0]]

    if orig_corners[1, sorted_x_id[2]] < orig_corners[1, sorted_x_id[3]]:
        new_corners[:, 1] = orig_corners[:, sorted_x_id[2]]
        new_corners[:, 2] = orig_corners[:, sorted_x_id[3]]
    else:
        new_corners[:, 1] = orig_corners[:, sorted_x_id[3]]
        new_corners[:, 2] = orig_corners[:, sorted_x_id[2]]
    return new_corners


def parseValue(old_val, new_val, arg_name=None):
    if type(old_val) is str:
        return new_val
    elif type(old_val) is int:
        return int(new_val)
    elif type(old_val) is float:
        return float(new_val)
    elif type(old_val) is tuple or type(old_val) is list:
        arg_list = new_val.split(',')
        if len(arg_list) != len(old_val):
            print('arg_list: ', arg_list)
            if arg_name is not None:
                raise SyntaxError('Invalid size for parameter {:s}: {:d}: '.format(
                    arg_name, len(arg_list)))
            else:
                raise SyntaxError('Invalid size for parameter: {:s}: '.format(len(arg_list)))
        for i in range(len(arg_list)):
            arg_list[i] = parseValue(old_val[i], arg_list[i])
        if type(old_val) is tuple:
            return tuple(arg_list)
        else:
            return arg_list


def parseArguments(args, params):
    print('args: \n', args)
    if (len(args) - 1) % 2 != 0:
        print('args: \n', args)
        raise SyntaxError('Command line arguments must be specified in pairs')
    arg_id = 1
    while arg_id < len(args):
        arg_name = args[arg_id]
        if not arg_name in params.keys():
            raise SyntaxError('Invalid command line argument: {:s}'.format(arg_name))
        params[arg_name] = parseValue(params[arg_name], args[arg_id + 1], arg_name)
        print('Setting ', arg_name, ' to ', params[arg_name])
        arg_id += 2
    return params


def write(str):
    sys.stdout.write(str)
    sys.stdout.flush()


def arrangeCornersWithIDs(orig_corners):
    # print 'orig_corners:\n', orig_corners
    # print 'orig_corners.shape:\n', orig_corners.shape

    new_corners = np.zeros((2, 4), dtype=np.float64)
    sorted_x_id = np.argsort(orig_corners[0, :])

    rearrangement_ids = np.array([0, 1, 2, 3], dtype=np.uint32)

    # print 'new_corners.shape:\n', new_corners.shape
    print('sorted_x_id:\n', sorted_x_id)

    if orig_corners[1, sorted_x_id[0]] < orig_corners[1, sorted_x_id[1]]:
        new_corners[:, 0] = orig_corners[:, sorted_x_id[0]]
        new_corners[:, 3] = orig_corners[:, sorted_x_id[1]]
        rearrangement_ids[0] = sorted_x_id[0]
        rearrangement_ids[3] = sorted_x_id[1]
    else:
        new_corners[:, 0] = orig_corners[:, sorted_x_id[1]]
        new_corners[:, 3] = orig_corners[:, sorted_x_id[0]]
        rearrangement_ids[0] = sorted_x_id[1]
        rearrangement_ids[3] = sorted_x_id[0]

    if orig_corners[1, sorted_x_id[2]] < orig_corners[1, sorted_x_id[3]]:
        new_corners[:, 1] = orig_corners[:, sorted_x_id[2]]
        new_corners[:, 2] = orig_corners[:, sorted_x_id[3]]
        rearrangement_ids[1] = sorted_x_id[2]
        rearrangement_ids[2] = sorted_x_id[3]
    else:
        new_corners[:, 1] = orig_corners[:, sorted_x_id[3]]
        new_corners[:, 2] = orig_corners[:, sorted_x_id[2]]
        rearrangement_ids[1] = sorted_x_id[3]
        rearrangement_ids[2] = sorted_x_id[2]
    return new_corners, rearrangement_ids


def add_suffix(src_path, suffix, sep='_'):
    # abs_src_path = os.path.abspath(src_path)
    src_dir = os.path.dirname(src_path)
    src_name, src_ext = os.path.splitext(os.path.basename(src_path))
    dst_path = os.path.join(src_dir, src_name + sep + suffix + src_ext)
    return dst_path


def getSyntheticSeqSuffix(syn_ssm, syn_ssm_sigma_id, syn_ilm='0',
                          syn_am_sigma_id=0, syn_add_noise=0,
                          syn_noise_mean=0, syn_noise_sigma=10):
    syn_out_suffix = 'warped_{:s}_s{:d}'.format(syn_ssm, syn_ssm_sigma_id)
    if syn_ilm != "0":
        syn_out_suffix = '{:s}_{:s}_s{:d}'.format(syn_out_suffix,
                                                  syn_ilm, syn_am_sigma_id)
    if syn_add_noise:
        syn_out_suffix = '{:s}_gauss_{:4.2f}_{:4.2f}'.format(syn_out_suffix,
                                                             syn_noise_mean, syn_noise_sigma)
    return syn_out_suffix


def getSyntheticSeqName(source_name, syn_ssm, syn_ssm_sigma_id, syn_ilm='0',
                        syn_am_sigma_id=0, syn_frame_id=0, syn_add_noise=0,
                        syn_noise_mean=0, syn_noise_sigma=10, syn_out_suffix=None):
    if syn_out_suffix is None:
        syn_out_suffix = getSyntheticSeqSuffix(syn_ssm, syn_ssm_sigma_id, syn_ilm,
                                               syn_am_sigma_id, syn_add_noise, syn_noise_mean, syn_noise_sigma)

    return '{:s}_{:d}_{:s}'.format(source_name, syn_frame_id, syn_out_suffix)


def getDateTime():
    return time.strftime("%y%m%d_%H%M", time.localtime())


class ParamDict:
    tracker_types = {0: 'gt',
                     1: 'esm',
                     2: 'ic',
                     3: 'nnic',
                     4: 'pf',
                     5: 'pw',
                     6: 'ppw'
                     }
    grid_types = {0: 'trans',
                  1: 'rs',
                  2: 'shear',
                  3: 'proj',
                  4: 'rtx',
                  5: 'rty',
                  6: 'stx',
                  7: 'sty',
                  8: 'trans2'
                  }
    filter_types = {0: 'none',
                    1: 'gauss',
                    2: 'box',
                    3: 'norm_box',
                    4: 'bilateral',
                    5: 'median',
                    6: 'gabor',
                    7: 'sobel',
                    8: 'scharr',
                    9: 'LoG',
                    10: 'DoG',
                    11: 'laplacian',
                    12: 'canny'
                    }
    inc_types = {0: 'fc',
                 1: 'ic',
                 2: 'fa',
                 3: 'ia'
                 }
    appearance_models = {0: 'ssd',
                         1: 'scv',
                         2: 'ncc',
                         3: 'mi',
                         4: 'ccre',
                         5: 'hssd',
                         6: 'jht',
                         7: 'mi2',
                         8: 'ncc2',
                         9: 'scv2',
                         10: 'mi_old',
                         11: 'mssd',
                         12: 'bmssd',
                         13: 'bmi',
                         14: 'crv',
                         15: 'fkld',
                         16: 'ikld',
                         17: 'mkld',
                         18: 'chis',
                         19: 'ssim'
                         }

    sequences_tmt = {
        0: 'nl_bookI_s3',
        1: 'nl_bookII_s3',
        2: 'nl_bookIII_s3',
        3: 'nl_cereal_s3',
        4: 'nl_juice_s3',
        5: 'nl_mugI_s3',
        6: 'nl_mugII_s3',
        7: 'nl_mugIII_s3',

        8: 'nl_bookI_s4',
        9: 'nl_bookII_s4',
        10: 'nl_bookIII_s4',
        11: 'nl_cereal_s4',
        12: 'nl_juice_s4',
        13: 'nl_mugI_s4',
        14: 'nl_mugII_s4',
        15: 'nl_mugIII_s4',

        16: 'nl_bus',
        17: 'nl_highlighting',
        18: 'nl_letter',
        19: 'nl_newspaper',

        20: 'nl_bookI_s1',
        21: 'nl_bookII_s1',
        22: 'nl_bookIII_s1',
        23: 'nl_cereal_s1',
        24: 'nl_juice_s1',
        25: 'nl_mugI_s1',
        26: 'nl_mugII_s1',
        27: 'nl_mugIII_s1',

        28: 'nl_bookI_s2',
        29: 'nl_bookII_s2',
        30: 'nl_bookIII_s2',
        31: 'nl_cereal_s2',
        32: 'nl_juice_s2',
        33: 'nl_mugI_s2',
        34: 'nl_mugII_s2',
        35: 'nl_mugIII_s2',

        36: 'nl_bookI_s5',
        37: 'nl_bookII_s5',
        38: 'nl_bookIII_s5',
        39: 'nl_cereal_s5',
        40: 'nl_juice_s5',
        41: 'nl_mugI_s5',
        42: 'nl_mugII_s5',
        43: 'nl_mugIII_s5',

        44: 'nl_bookI_si',
        45: 'nl_bookII_si',
        46: 'nl_cereal_si',
        47: 'nl_juice_si',
        48: 'nl_mugI_si',
        49: 'nl_mugIII_si',

        50: 'dl_bookI_s3',
        51: 'dl_bookII_s3',
        52: 'dl_bookIII_s3',
        53: 'dl_cereal_s3',
        54: 'dl_juice_s3',
        55: 'dl_mugI_s3',
        56: 'dl_mugII_s3',
        57: 'dl_mugIII_s3',

        58: 'dl_bookI_s4',
        59: 'dl_bookII_s4',
        60: 'dl_bookIII_s4',
        61: 'dl_cereal_s4',
        62: 'dl_juice_s4',
        63: 'dl_mugI_s4',
        64: 'dl_mugII_s4',
        65: 'dl_mugIII_s4',

        66: 'dl_bus',
        67: 'dl_highlighting',
        68: 'dl_letter',
        69: 'dl_newspaper',

        70: 'dl_bookI_s1',
        71: 'dl_bookII_s1',
        72: 'dl_bookIII_s1',
        73: 'dl_cereal_s1',
        74: 'dl_juice_s1',
        75: 'dl_mugI_s1',
        76: 'dl_mugII_s1',
        77: 'dl_mugIII_s1',

        78: 'dl_bookI_s2',
        79: 'dl_bookII_s2',
        80: 'dl_bookIII_s2',
        81: 'dl_cereal_s2',
        82: 'dl_juice_s2',
        83: 'dl_mugI_s2',
        84: 'dl_mugII_s2',
        85: 'dl_mugIII_s2',

        86: 'dl_bookI_s5',
        87: 'dl_bookII_s5',
        88: 'dl_bookIII_s5',
        89: 'dl_cereal_s5',
        90: 'dl_juice_s5',
        91: 'dl_mugI_s5',
        92: 'dl_mugIII_s5',

        93: 'dl_bookII_si',
        94: 'dl_cereal_si',
        95: 'dl_juice_si',
        96: 'dl_mugI_si',
        97: 'dl_mugIII_si',

        98: 'dl_mugII_si',
        99: 'dl_mugII_s5',
        100: 'nl_mugII_si',

        101: 'robot_bookI',
        102: 'robot_bookII',
        103: 'robot_bookIII',
        104: 'robot_cereal',
        105: 'robot_juice',
        106: 'robot_mugI',
        107: 'robot_mugII',
        108: 'robot_mugIII'
    }
    sequences_ucsb = {
        0: 'bricks_dynamic_lighting',
        1: 'bricks_motion1',
        2: 'bricks_motion2',
        3: 'bricks_motion3',
        4: 'bricks_motion4',
        5: 'bricks_motion5',
        6: 'bricks_motion6',
        7: 'bricks_motion7',
        8: 'bricks_motion8',
        9: 'bricks_motion9',
        10: 'bricks_panning',
        11: 'bricks_perspective',
        12: 'bricks_rotation',
        13: 'bricks_static_lighting',
        14: 'bricks_unconstrained',
        15: 'bricks_zoom',
        16: 'building_dynamic_lighting',
        17: 'building_motion1',
        18: 'building_motion2',
        19: 'building_motion3',
        20: 'building_motion4',
        21: 'building_motion5',
        22: 'building_motion6',
        23: 'building_motion7',
        24: 'building_motion8',
        25: 'building_motion9',
        26: 'building_panning',
        27: 'building_perspective',
        28: 'building_rotation',
        29: 'building_static_lighting',
        30: 'building_unconstrained',
        31: 'building_zoom',
        32: 'mission_dynamic_lighting',
        33: 'mission_motion1',
        34: 'mission_motion2',
        35: 'mission_motion3',
        36: 'mission_motion4',
        37: 'mission_motion5',
        38: 'mission_motion6',
        39: 'mission_motion7',
        40: 'mission_motion8',
        41: 'mission_motion9',
        42: 'mission_panning',
        43: 'mission_perspective',
        44: 'mission_rotation',
        45: 'mission_static_lighting',
        46: 'mission_unconstrained',
        47: 'mission_zoom',
        48: 'paris_dynamic_lighting',
        49: 'paris_motion1',
        50: 'paris_motion2',
        51: 'paris_motion3',
        52: 'paris_motion4',
        53: 'paris_motion5',
        54: 'paris_motion6',
        55: 'paris_motion7',
        56: 'paris_motion8',
        57: 'paris_motion9',
        58: 'paris_panning',
        59: 'paris_perspective',
        60: 'paris_rotation',
        61: 'paris_static_lighting',
        62: 'paris_unconstrained',
        63: 'paris_zoom',
        64: 'sunset_dynamic_lighting',
        65: 'sunset_motion1',
        66: 'sunset_motion2',
        67: 'sunset_motion3',
        68: 'sunset_motion4',
        69: 'sunset_motion5',
        70: 'sunset_motion6',
        71: 'sunset_motion7',
        72: 'sunset_motion8',
        73: 'sunset_motion9',
        74: 'sunset_panning',
        75: 'sunset_perspective',
        76: 'sunset_rotation',
        77: 'sunset_static_lighting',
        78: 'sunset_unconstrained',
        79: 'sunset_zoom',
        80: 'wood_dynamic_lighting',
        81: 'wood_motion1',
        82: 'wood_motion2',
        83: 'wood_motion3',
        84: 'wood_motion4',
        85: 'wood_motion5',
        86: 'wood_motion6',
        87: 'wood_motion7',
        88: 'wood_motion8',
        89: 'wood_motion9',
        90: 'wood_panning',
        91: 'wood_perspective',
        92: 'wood_rotation',
        93: 'wood_static_lighting',
        94: 'wood_unconstrained',
        95: 'wood_zoom'
    }
    sequences_lintrack = {
        0: 'mouse_pad',
        1: 'phone',
        2: 'towel',
    }
    sequences_lintrack_short = {
        0: 'mouse_pad_1',
        1: 'mouse_pad_2',
        2: 'mouse_pad_3',
        3: 'mouse_pad_4',
        4: 'mouse_pad_5',
        5: 'mouse_pad_6',
        6: 'mouse_pad_7',
        7: 'phone_1',
        8: 'phone_2',
        9: 'phone_3',
        10: 'towel_1',
        11: 'towel_2',
        12: 'towel_3',
        13: 'towel_4',
    }
    sequences_pami = {
        0: 'acronis',
        1: 'bass',
        2: 'bear',
        3: 'board_robot',
        4: 'board_robot_2',
        5: 'book1',
        6: 'book2',
        7: 'book3',
        8: 'book4',
        9: 'box',
        10: 'box_robot',
        11: 'cat_cylinder',
        12: 'cat_mask',
        13: 'cat_plane',
        14: 'compact_disc',
        15: 'cube',
        16: 'dft_atlas_moving',
        17: 'dft_atlas_still',
        18: 'dft_moving',
        19: 'dft_still',
        20: 'juice',
        21: 'lemming',
        22: 'mascot',
        23: 'omni_magazine',
        24: 'omni_obelix',
        25: 'sylvester',
        26: 'table_top',
        27: 'tea'
    }
    sequences_ptw = {
        0: 'Amish_1',
        1: 'Amish_2',
        2: 'Amish_3',
        3: 'Amish_4',
        4: 'Amish_5',
        5: 'Amish_6',
        6: 'Amish_7',
        7: 'Burger_1',
        8: 'Burger_2',
        9: 'Burger_3',
        10: 'Burger_4',
        11: 'Burger_5',
        12: 'Burger_6',
        13: 'Burger_7',
        14: 'BusStop_1',
        15: 'BusStop_2',
        16: 'BusStop_3',
        17: 'BusStop_4',
        18: 'BusStop_5',
        19: 'BusStop_6',
        20: 'BusStop_7',
        21: 'Citibank_1',
        22: 'Citibank_2',
        23: 'Citibank_3',
        24: 'Citibank_4',
        25: 'Citibank_5',
        26: 'Citibank_6',
        27: 'Citibank_7',
        28: 'Coke_1',
        29: 'Coke_2',
        30: 'Coke_3',
        31: 'Coke_4',
        32: 'Coke_5',
        33: 'Coke_6',
        34: 'Coke_7',
        35: 'Fruit_1',
        36: 'Fruit_2',
        37: 'Fruit_3',
        38: 'Fruit_4',
        39: 'Fruit_5',
        40: 'Fruit_6',
        41: 'Fruit_7',
        42: 'IndegoStation_1',
        43: 'IndegoStation_2',
        44: 'IndegoStation_3',
        45: 'IndegoStation_4',
        46: 'IndegoStation_5',
        47: 'IndegoStation_6',
        48: 'IndegoStation_7',
        49: 'Lottery_1_1',
        50: 'Lottery_1_2',
        51: 'Lottery_1_3',
        52: 'Lottery_1_4',
        53: 'Lottery_1_5',
        54: 'Lottery_1_6',
        55: 'Lottery_1_7',
        56: 'Lottery_2_1',
        57: 'Lottery_2_2',
        58: 'Lottery_2_3',
        59: 'Lottery_2_4',
        60: 'Lottery_2_5',
        61: 'Lottery_2_6',
        62: 'Lottery_2_7',
        63: 'Map_1_1',
        64: 'Map_1_2',
        65: 'Map_1_3',
        66: 'Map_1_4',
        67: 'Map_1_5',
        68: 'Map_1_6',
        69: 'Map_1_7',
        70: 'Map_2_1',
        71: 'Map_2_2',
        72: 'Map_2_3',
        73: 'Map_2_4',
        74: 'Map_2_5',
        75: 'Map_2_6',
        76: 'Map_2_7',
        77: 'Map_3_1',
        78: 'Map_3_2',
        79: 'Map_3_3',
        80: 'Map_3_4',
        81: 'Map_3_5',
        82: 'Map_3_6',
        83: 'Map_3_7',
        84: 'Melts_1',
        85: 'Melts_2',
        86: 'Melts_3',
        87: 'Melts_4',
        88: 'Melts_5',
        89: 'Melts_6',
        90: 'Melts_7',
        91: 'NoStopping_1',
        92: 'NoStopping_2',
        93: 'NoStopping_3',
        94: 'NoStopping_4',
        95: 'NoStopping_5',
        96: 'NoStopping_6',
        97: 'NoStopping_7',
        98: 'OneWay_1',
        99: 'OneWay_2',
        100: 'OneWay_3',
        101: 'OneWay_4',
        102: 'OneWay_5',
        103: 'OneWay_6',
        104: 'OneWay_7',
        105: 'Painting_1_1',
        106: 'Painting_1_2',
        107: 'Painting_1_3',
        108: 'Painting_1_4',
        109: 'Painting_1_5',
        110: 'Painting_1_6',
        111: 'Painting_1_7',
        112: 'Painting_2_1',
        113: 'Painting_2_2',
        114: 'Painting_2_3',
        115: 'Painting_2_4',
        116: 'Painting_2_5',
        117: 'Painting_2_6',
        118: 'Painting_2_7',
        119: 'Pizza_1',
        120: 'Pizza_2',
        121: 'Pizza_3',
        122: 'Pizza_4',
        123: 'Pizza_5',
        124: 'Pizza_6',
        125: 'Pizza_7',
        126: 'Poster_1_1',
        127: 'Poster_1_2',
        128: 'Poster_1_3',
        129: 'Poster_1_4',
        130: 'Poster_1_5',
        131: 'Poster_1_6',
        132: 'Poster_1_7',
        133: 'Poster_2_1',
        134: 'Poster_2_2',
        135: 'Poster_2_3',
        136: 'Poster_2_4',
        137: 'Poster_2_5',
        138: 'Poster_2_6',
        139: 'Poster_2_7',
        140: 'Pretzel_1',
        141: 'Pretzel_2',
        142: 'Pretzel_3',
        143: 'Pretzel_4',
        144: 'Pretzel_5',
        145: 'Pretzel_6',
        146: 'Pretzel_7',
        147: 'ShuttleStop_1',
        148: 'ShuttleStop_2',
        149: 'ShuttleStop_3',
        150: 'ShuttleStop_4',
        151: 'ShuttleStop_5',
        152: 'ShuttleStop_6',
        153: 'ShuttleStop_7',
        154: 'SmokeFree_1',
        155: 'SmokeFree_2',
        156: 'SmokeFree_3',
        157: 'SmokeFree_4',
        158: 'SmokeFree_5',
        159: 'SmokeFree_6',
        160: 'SmokeFree_7',
        161: 'Snack_1',
        162: 'Snack_2',
        163: 'Snack_3',
        164: 'Snack_4',
        165: 'Snack_5',
        166: 'Snack_6',
        167: 'Snack_7',
        168: 'Snap_1',
        169: 'Snap_2',
        170: 'Snap_3',
        171: 'Snap_4',
        172: 'Snap_5',
        173: 'Snap_6',
        174: 'Snap_7',
        175: 'StopSign_1',
        176: 'StopSign_2',
        177: 'StopSign_3',
        178: 'StopSign_4',
        179: 'StopSign_5',
        180: 'StopSign_6',
        181: 'StopSign_7',
        182: 'Sundae_1',
        183: 'Sundae_2',
        184: 'Sundae_3',
        185: 'Sundae_4',
        186: 'Sundae_5',
        187: 'Sundae_6',
        188: 'Sundae_7',
        189: 'Sunoco_1',
        190: 'Sunoco_2',
        191: 'Sunoco_3',
        192: 'Sunoco_4',
        193: 'Sunoco_5',
        194: 'Sunoco_6',
        195: 'Sunoco_7',
        196: 'WalkYourBike_1',
        197: 'WalkYourBike_2',
        198: 'WalkYourBike_3',
        199: 'WalkYourBike_4',
        200: 'WalkYourBike_5',
        201: 'WalkYourBike_6',
        202: 'WalkYourBike_7',
        203: 'Woman_1',
        204: 'Woman_2',
        205: 'Woman_3',
        206: 'Woman_4',
        207: 'Woman_5',
        208: 'Woman_6',
        209: 'Woman_7'
    }
    sequences_cmt = {
        0: 'board_robot',
        1: 'box_robot',
        2: 'cup_on_table',
        3: 'juice',
        4: 'lemming',
        5: 'liquor',
        6: 'sylvester',
        7: 'ball',
        8: 'car',
        9: 'car_2',
        10: 'carchase',
        11: 'dog1',
        12: 'gym',
        13: 'jumping',
        14: 'mountain_bike',
        15: 'person',
        16: 'person_crossing',
        17: 'person_partially_occluded',
        18: 'singer',
        19: 'track_running'
    }

    sequences_vivid = {
        0: 'redteam',
        1: 'egtest01',
        2: 'egtest02',
        3: 'egtest03',
        4: 'egtest04',
        5: 'egtest05',
        6: 'pktest01',
        7: 'pktest02',
        8: 'pktest03'
    }

    sequences_trakmark = {
        0: 'CV00_00',
        1: 'CV00_01',
        2: 'CV00_02',
        3: 'CV01_00',
        4: 'FS00_00',
        5: 'FS00_01',
        6: 'FS00_02',
        7: 'FS00_03',
        8: 'FS00_04',
        9: 'FS00_05',
        10: 'FS00_06',
        11: 'FS01_00',
        12: 'FS01_01',
        13: 'FS01_02',
        14: 'FS01_03',
        15: 'JR00_00',
        16: 'JR00_01',
        17: 'NC00_00',
        18: 'NC01_00',
        19: 'NH00_00',
        20: 'NH00_01'
    }

    sequences_vot = {
        0: 'woman',
        1: 'ball',
        2: 'basketball',
        3: 'bicycle',
        4: 'bolt',
        5: 'car',
        6: 'david',
        7: 'diving',
        8: 'drunk',
        9: 'fernando',
        10: 'fish1',
        11: 'fish2',
        12: 'gymnastics',
        13: 'hand1',
        14: 'hand2',
        15: 'jogging',
        16: 'motocross',
        17: 'polarbear',
        18: 'skating',
        19: 'sphere',
        20: 'sunshade',
        21: 'surfing',
        22: 'torus',
        23: 'trellis',
        24: 'tunnel'
    }

    sequences_vot16 = {
        0: 'bag',
        1: 'ball1',
        2: 'ball2',
        3: 'basketball',
        4: 'birds1',
        5: 'birds2',
        6: 'blanket',
        7: 'bmx',
        8: 'bolt1',
        9: 'bolt2',
        10: 'book',
        11: 'butterfly',
        12: 'car1',
        13: 'car2',
        14: 'crossing',
        15: 'dinosaur',
        16: 'fernando',
        17: 'fish1',
        18: 'fish2',
        19: 'fish3',
        20: 'fish4',
        21: 'girl',
        22: 'glove',
        23: 'godfather',
        24: 'graduate',
        25: 'gymnastics1',
        26: 'gymnastics2',
        27: 'gymnastics3',
        28: 'gymnastics4',
        29: 'hand',
        30: 'handball1',
        31: 'handball2',
        32: 'helicopter',
        33: 'iceskater1',
        34: 'iceskater2',
        35: 'leaves',
        36: 'marching',
        37: 'matrix',
        38: 'motocross1',
        39: 'motocross2',
        40: 'nature',
        41: 'octopus',
        42: 'pedestrian1',
        43: 'pedestrian2',
        44: 'rabbit',
        45: 'racing',
        46: 'road',
        47: 'shaking',
        48: 'sheep',
        49: 'singer1',
        50: 'singer2',
        51: 'singer3',
        52: 'soccer1',
        53: 'soccer2',
        54: 'soldier',
        55: 'sphere',
        56: 'tiger',
        57: 'traffic',
        58: 'tunnel',
        59: 'wiper'
    }

    sequences_vtb = {
        0: 'Basketball',
        1: 'Biker',
        2: 'Bird1',
        3: 'Bird2',
        4: 'BlurBody',
        5: 'BlurCar1',
        6: 'BlurCar2',
        7: 'BlurCar3',
        8: 'BlurCar4',
        9: 'BlurFace',
        10: 'BlurOwl',
        11: 'Board',
        12: 'Bolt',
        13: 'Bolt2',
        14: 'Box',
        15: 'Boy',
        16: 'Car1',
        17: 'Car2',
        18: 'Car4',
        19: 'Car24',
        20: 'CarDark',
        21: 'CarScale',
        22: 'ClifBar',
        23: 'Coke',
        24: 'Couple',
        25: 'Coupon',
        26: 'Crossing',
        27: 'Crowds',
        28: 'Dancer',
        29: 'Dancer2',
        30: 'David',
        31: 'David2',
        32: 'David3',
        33: 'Deer',
        34: 'Diving',
        35: 'Dog',
        36: 'Dog1',
        37: 'Doll',
        38: 'DragonBaby',
        39: 'Dudek',
        40: 'FaceOcc1',
        41: 'FaceOcc2',
        42: 'Fish',
        43: 'FleetFace',
        44: 'Football',
        45: 'Football1',
        46: 'Freeman1',
        47: 'Freeman3',
        48: 'Freeman4',
        49: 'Girl',
        50: 'Girl2',
        51: 'Gym',
        52: 'Human2',
        53: 'Human3',
        54: 'Human4',
        55: 'Human5',
        56: 'Human6',
        57: 'Human7',
        58: 'Human8',
        59: 'Human9',
        60: 'Ironman',
        61: 'Jogging',
        62: 'Jogging_2',
        63: 'Jump',
        64: 'Jumping',
        65: 'KiteSurf',
        66: 'Lemming',
        67: 'Liquor',
        68: 'Man',
        69: 'Matrix',
        70: 'Mhyang',
        71: 'MotorRolling',
        72: 'MountainBike',
        73: 'Panda',
        74: 'RedTeam',
        75: 'Rubik',
        76: 'Shaking',
        77: 'Singer1',
        78: 'Singer2',
        79: 'Skater',
        80: 'Skater2',
        81: 'Skating1',
        82: 'Skating2',
        83: 'Skating2_2',
        84: 'Skiing',
        85: 'Soccer',
        86: 'Subway',
        87: 'Surfer',
        88: 'Suv',
        89: 'Sylvester',
        90: 'Tiger1',
        91: 'Tiger2',
        92: 'Toy',
        93: 'Trans',
        94: 'Trellis',
        95: 'Twinnings',
        96: 'Vase',
        97: 'Walking',
        98: 'Walking2',
        99: 'Woman'
    }

    sequences_metaio = {
        0: 'bump_angle',
        1: 'bump_fast_close',
        2: 'bump_fast_far',
        3: 'bump_illumination',
        4: 'bump_range',
        5: 'grass_angle',
        6: 'grass_fast_close',
        7: 'grass_fast_far',
        8: 'grass_illumination',
        9: 'grass_range',
        10: 'isetta_angle',
        11: 'isetta_fast_close',
        12: 'isetta_fast_far',
        13: 'isetta_illumination',
        14: 'isetta_range',
        15: 'lucent_angle',
        16: 'lucent_fast_close',
        17: 'lucent_fast_far',
        18: 'lucent_illumination',
        19: 'lucent_range',
        20: 'macMini_angle',
        21: 'macMini_fast_close',
        22: 'macMini_fast_far',
        23: 'macMini_illumination',
        24: 'macMini_range',
        25: 'philadelphia_angle',
        26: 'philadelphia_fast_close',
        27: 'philadelphia_fast_far',
        28: 'philadelphia_illumination',
        29: 'philadelphia_range',
        30: 'stop_angle',
        31: 'stop_fast_close',
        32: 'stop_fast_far',
        33: 'stop_illumination',
        34: 'stop_range',
        35: 'wall_angle',
        36: 'wall_fast_close',
        37: 'wall_fast_far',
        38: 'wall_illumination',
        39: 'wall_range'
    }

    sequences_tfmt = {
        0: 'fish_lure_left',
        1: 'fish_lure_right',
        2: 'fish_lure_fast_left',
        3: 'fish_lure_fast_right',
        4: 'key_task_left',
        5: 'key_task_right',
        6: 'key_task_fast_left',
        7: 'key_task_fast_right',
        8: 'hexagon_task_left',
        9: 'hexagon_task_right',
        10: 'hexagon_task_fast_left',
        11: 'hexagon_task_fast_right',
        12: 'fish_lure_cam1',
        13: 'fish_lure_cam2',
        14: 'fish_lure_fast_cam1',
        15: 'fish_lure_fast_cam2',
        16: 'key_task_cam1',
        17: 'key_task_cam2',
        18: 'key_task_fast_cam1',
        19: 'key_task_fast_cam2',
        20: 'hexagon_task_cam1',
        21: 'hexagon_task_cam2',
        22: 'hexagon_task_fast_cam1',
        23: 'hexagon_task_fast_cam2'
    }

    sequences_tmt_fine_full = {
        0: 'fish_lure_left',
        1: 'fish_lure_right',
        2: 'fish_lure_fast_left',
        3: 'fish_lure_fast_right',
        4: 'key_task_left',
        5: 'key_task_right',
        6: 'key_task_fast_left',
        7: 'key_task_fast_right',
        8: 'hexagon_task_left',
        9: 'hexagon_task_right',
        10: 'hexagon_task_fast_left',
        11: 'hexagon_task_fast_right',
        12: 'fish_lure_cam1',
        13: 'fish_lure_cam2',
        14: 'fish_lure_fast_cam1',
        15: 'fish_lure_fast_cam2',
        16: 'key_task_cam1',
        17: 'key_task_cam2',
        18: 'key_task_fast_cam1',
        19: 'key_task_fast_cam2',
        20: 'hexagon_task_cam1',
        21: 'hexagon_task_cam2',
        22: 'hexagon_task_fast_cam1',
        23: 'hexagon_task_fast_cam2',
    }

    sequences_mosaic = {
        0: 'book_1',
        1: 'book_2',
        2: 'book_3',
        3: 'book_4',
        4: 'book_5',
        5: 'book_6',
        6: 'book_7',
        7: 'book_8',
        8: 'poster_1',
        9: 'poster_2',
        10: 'poster_3',
        11: 'poster_4',
        12: 'poster_5',
        13: 'poster_6',
        14: 'poster_7',
        15: 'poster_8',
        16: 'poster_9'
    }

    sequences_misc = {
        0: 'uav_sim',
        1: 'chess_board_1',
        2: 'chess_board_2',
        3: 'chess_board_3',
        4: 'chess_board_4'
    }
    sequences_synthetic = {
        0: 'bear',
        1: 'board_robot',
        2: 'book4',
        3: 'box',
        4: 'box_robot',
        5: 'building_dynamic_lighting',
        6: 'cat_cylinder',
        7: 'cube',
        8: 'dft_still',
        9: 'lemming',
        10: 'mission_dynamic_lighting',
        11: 'mouse_pad',
        12: 'nl_bookI_s3',
        13: 'nl_bus',
        14: 'nl_cereal_s3',
        15: 'nl_juice_s3',
        16: 'nl_letter',
        17: 'nl_mugI_s3',
        18: 'nl_newspaper',
        19: 'paris_dynamic_lighting',
        20: 'phone',
        21: 'sunset_dynamic_lighting',
        22: 'sylvester',
        23: 'towel',
        24: 'wood_dynamic_lighting'
    }

    sequences_live = {
        0: 'usb_cam',
        1: 'firewire_cam'
    }
    actors = {
        0: 'TMT',
        1: 'UCSB',
        2: 'LinTrack',
        3: 'PAMI',
        4: 'TMT_FINE',
        5: 'PTW',
        6: 'METAIO',
        7: 'CMT',
        8: 'VOT',
        9: 'VOT16',
        10: 'VTB',
        11: 'VIVID',
        12: 'TrakMark',
        13: 'LinTrackShort',
        14: 'Mosaic',
        15: 'Misc',
        16: 'Synthetic',
        17: 'Live'
    }
    # sequences = dict(zip([actors[i] for i in range(len(actors))],
    #                      [sequences_tmt,
    #                       sequences_ucsb,
    #                       sequences_lintrack,
    #                       sequences_pami,
    #                       sequences_tfmt,
    #                       sequences_ptw,
    #                       sequences_metaio,
    #                       sequences_cmt,
    #                       sequences_vot,
    #                       sequences_vot16,
    #                       sequences_vtb,
    #                       sequences_vivid,
    #                       sequences_trakmark,
    #                       sequences_lintrack_short,
    #                       sequences_mosaic,
    #                       sequences_misc,
    #                       sequences_synthetic,
    #                       sequences_live]))

    sequences_mot2015_train = {
        0: 'TUD-Stadtmitte',
        1: 'TUD-Campus',
        2: 'PETS09-S2L1',
        3: 'ETH-Bahnhof',
        4: 'ETH-Sunnyday',
        5: 'ETH-Pedcross2',
        6: 'ADL-Rundle-6',
        7: 'ADL-Rundle-8',
        8: 'KITTI-13',
        9: 'KITTI-17',
        10: 'Venice-2'
    }
    sequences_mot2015_test = {
        0: 'TUD-Crossing',
        1: 'PETS09-S2L2',
        2: 'ETH-Jelmoli',
        3: 'ETH-Linthescher',
        4: 'ETH-Crossing',
        5: 'AVG-TownCentre',
        6: 'ADL-Rundle-1',
        7: 'ADL-Rundle-3',
        8: 'KITTI-16',
        9: 'KITTI-19',
        10: 'Venice-1'
    }
    sequences_mot2017_train = {
        0: 'TUD-Stadtmitte',
        1: 'TUD-Campus',
        2: 'PETS09-S2L1',
        3: 'ETH-Bahnhof',
        4: 'ETH-Sunnyday',
        5: 'ETH-Pedcross2',
        6: 'ADL-Rundle-6',
        7: 'ADL-Rundle-8',
        8: 'KITTI-13',
        9: 'KITTI-17',
        10: 'Venice-2'
    }
    sequences_mot2017_test = {
        0: 'TUD-Crossing',
        1: 'PETS09-S2L2',
        2: 'ETH-Jelmoli',
        3: 'ETH-Linthescher',
        4: 'ETH-Crossing',
        5: 'AVG-TownCentre',
        6: 'ADL-Rundle-1',
        7: 'ADL-Rundle-3',
        8: 'KITTI-16',
        9: 'KITTI-19',
        10: 'Venice-1'
    }
    sequences_kitti_train = {
        0: '0000',
        1: '0001',
        2: '0002',
        3: '0003',
        4: '0004',
        5: '0005',
        6: '0006',
        7: '0007',
        8: '0008',
        9: '0009',
        10: '0010',
        11: '0011',
        12: '0012',
        13: '0013',
        14: '0014',
        15: '0015',
        16: '0016',
        17: '0017',
        18: '0018',
        19: '0019',
        20: '0020'
    }
    sequences_kitti_test = {
        0: '0000',
        1: '0001',
        2: '0002',
        3: '0003',
        4: '0004',
        5: '0005',
        6: '0006',
        7: '0007',
        8: '0008',
        9: '0009',
        10: '0010',
        11: '0011',
        12: '0012',
        13: '0013',
        14: '0014',
        15: '0015',
        16: '0016',
        17: '0017',
        18: '0018',
        19: '0019',
        20: '0020',
        21: '0021',
        22: '0022',
        23: '0023',
        24: '0024',
        25: '0025',
        26: '0026',
        27: '0027',
        28: '0028'
    }
    sequences_gram = {
        0: 'M-30',
        1: 'M-30-HD',
        2: 'Urban1',
        3: 'M-30-Large',
        4: 'M-30-HD-Small'
    }
    sequences_idot = {
        0: 'seq_1',
        1: 'seq_2',
        2: 'seq_3',
        3: 'seq_4',
        4: 'seq_5',
        5: 'seq_6',
        6: 'seq_7',
        7: 'seq_8',
        8: 'seq_9',
        9: 'seq_10',
        10: 'seq_11',
        11: 'seq_12',
        12: 'seq_13'
    }

    sequences_detrac = [
        'MVI_20011',
        'MVI_20012',
        'MVI_20032',
        'MVI_20033',
        'MVI_20034',
        'MVI_20035',
        'MVI_20051',
        'MVI_20052',
        'MVI_20061',
        'MVI_20062',
        'MVI_20063',
        'MVI_20064',
        'MVI_20065',
        'MVI_39761',
        'MVI_39771',
        'MVI_39781',
        'MVI_39801',
        'MVI_39811',
        'MVI_39821',
        'MVI_39851',
        'MVI_39861',
        'MVI_39931',
        'MVI_40131',
        'MVI_40141',
        'MVI_40152',
        'MVI_40161',
        'MVI_40162',
        'MVI_40171',
        'MVI_40172',
        'MVI_40181',
        'MVI_40191',
        'MVI_40192',
        'MVI_40201',
        'MVI_40204',
        'MVI_40211',
        'MVI_40212',
        'MVI_40213',
        'MVI_40241',
        'MVI_40243',
        'MVI_40244',
        'MVI_40732',
        'MVI_40751',
        'MVI_40752',
        'MVI_40871',
        'MVI_40962',
        'MVI_40963',
        'MVI_40981',
        'MVI_40991',
        'MVI_40992',
        'MVI_41063',
        'MVI_41073',
        'MVI_63521',
        'MVI_63525',
        'MVI_63544',
        'MVI_63552',
        'MVI_63553',
        'MVI_63554',
        'MVI_63561',
        'MVI_63562',
        'MVI_63563',

        'MVI_39031',
        'MVI_39051',
        'MVI_39211',
        'MVI_39271',
        'MVI_39311',
        'MVI_39361',
        'MVI_39371',
        'MVI_39401',
        'MVI_39501',
        'MVI_39511',
        'MVI_40701',
        'MVI_40711',
        'MVI_40712',
        'MVI_40714',
        'MVI_40742',
        'MVI_40743',
        'MVI_40761',
        'MVI_40762',
        'MVI_40763',
        'MVI_40771',
        'MVI_40772',
        'MVI_40773',
        'MVI_40774',
        'MVI_40775',
        'MVI_40792',
        'MVI_40793',
        'MVI_40851',
        'MVI_40852',
        'MVI_40853',
        'MVI_40854',
        'MVI_40855',
        'MVI_40863',
        'MVI_40864',
        'MVI_40891',
        'MVI_40892',
        'MVI_40901',
        'MVI_40902',
        'MVI_40903',
        'MVI_40904',
        'MVI_40905',
    ]
    mot_actors = {
        0: 'MOT2015',
        1: 'MOT2017',
        2: 'KITTI',
        3: 'GRAM',
        4: 'IDOT',
        5: 'DETRAC',
        6: 'CTC',
        # 6: 'DETRAC_Test',
    }

    from collections import OrderedDict

    sequences_ctc = OrderedDict({
        # train
        # 'dummy_01': 40,
        'BF-C2DL-HSC_01': 40,
        'BF-C2DL-HSC_02': 40,
        'BF-C2DL-MuSC_01': 40,
        'BF-C2DL-MuSC_02': 40,
        'DIC-C2DH-HeLa_01': 40,

        'DIC-C2DH-HeLa_02': 40,
        'Fluo-C2DL-Huh7_01': 40,
        'Fluo-C2DL-Huh7_02': 40,
        'Fluo-C2DL-MSC_01': 40,
        'Fluo-C2DL-MSC_02': 40,

        'Fluo-N2DH-GOWT1_01': 40,
        'Fluo-N2DH-GOWT1_02': 40,
        'Fluo-N2DL-HeLa_01': 40,
        'Fluo-N2DL-HeLa_02': 40,
        'Fluo-N2DH-SIM_01': 40,
        'Fluo-N2DH-SIM_02': 40,
        'PhC-C2DH-U373_01': 40,
        'PhC-C2DH-U373_02': 40,
        'PhC-C2DL-PSC_01': 40,
        'PhC-C2DL-PSC_02': 40,

        # test
        'BF-C2DL-HSC_Test_01': 40,
        'BF-C2DL-HSC_Test_02': 40,
        'BF-C2DL-MuSC_Test_01': 40,
        'BF-C2DL-MuSC_Test_02': 40,
        'DIC-C2DH-HeLa_Test_01': 40,
        'DIC-C2DH-HeLa_Test_02': 40,
        'Fluo-C2DL-Huh7_Test_01': 40,
        'Fluo-C2DL-Huh7_Test_02': 40,
        'Fluo-C2DL-MSC_Test_01': 40,
        'Fluo-C2DL-MSC_Test_02': 40,
        'Fluo-N2DH-GOWT1_Test_01': 40,
        'Fluo-N2DH-GOWT1_Test_02': 40,
        'Fluo-N2DL-HeLa_Test_01': 40,
        'Fluo-N2DL-HeLa_Test_02': 40,
        'Fluo-N2DH-SIM_Test_01': 40,
        'Fluo-N2DH-SIM_Test_02': 40,
        'PhC-C2DH-U373_Test_01': 40,
        'PhC-C2DH-U373_Test_02': 40,
        'PhC-C2DL-PSC_Test_01': 40,
        'PhC-C2DL-PSC_Test_02': 40,
    }
    )
    # mot_sequences = dict(zip([mot_actors[i] for i in range(len(mot_actors))], [
    #     [sequences_mot2015_train, sequences_mot2015_test],
    #     [sequences_mot2017_train, sequences_mot2017_test],
    #     [sequences_kitti_train, sequences_kitti_test],
    #     sequences_gram,
    #     sequences_idot,
    #     sequences_detrac,
    #     sequences_ctc,
    #     # sequences_detrac_test,
    # ]))

    challenges = {0: 'angle',
                  1: 'fast_close',
                  2: 'fast_far',
                  3: 'range',
                  4: 'illumination'
                  }
    opt_types = {0: 'pre',
                 1: 'post',
                 2: 'ind'
                 }
    mtf_sms = {0: 'esm',
               1: 'nesm',
               2: 'aesm',
               3: 'fclk',
               4: 'iclk',
               5: 'falk',
               6: 'ialk',
               7: 'nnic',
               8: 'casc',
               9: 'dsst',
               10: 'tld',
               11: 'kcf',
               12: 'pf'
               }
    mtf_ams = {0: 'ssd',
               1: 'ncc',
               2: 'scv',
               3: 'rscv',
               4: 'nssd',
               5: 'mi'
               }
    mtf_ssms = {0: '2',
                1: '3',
                2: '3s',
                3: '4',
                4: '6',
                5: '8',
                6: 'c8',
                7: 'l8'
                }
    opt_methods = {
        0: 'Newton-CG',
        1: 'CG',
        2: 'BFGS',
        3: 'Nelder-Mead',
        4: 'Powell',
        5: 'dogleg',
        6: 'trust-ncg',
        7: 'L-BFGS-B',
        8: 'TNC',
        9: 'COBYLA',
        10: 'SLSQP'
    }
    # params_dict = {'actors': actors,
    #                'sequences': sequences,
    #                'mot_actors': mot_actors,
    #                'mot_sequences': mot_sequences,
    #                'tracker_types': tracker_types,
    #                'grid_types': grid_types,
    #                'filter_types': filter_types,
    #                'inc_types': inc_types,
    #                'appearance_models': appearance_models,
    #                'opt_types': opt_types,
    #                'challenges': challenges,
    #                'mtf_sms': mtf_sms,
    #                'mtf_ams': mtf_ams,
    #                'mtf_ssms': mtf_ssms,
    #                'opt_methods': opt_methods
    #                }
    # return params_dict


def is_square(apositiveint):
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen: return False
        seen.add(x)
    return True


def addBorder(img, border_size, border_type):
    img_h, img_w = img.shape[:2]
    out_img_h, out_img_w = img_h, img_w
    start_row = start_col = 0
    if border_type == 'top':
        out_img_h += border_size
        start_row += border_size
    elif border_type == 'bottom':
        out_img_h += border_size
    elif border_type == 'left':
        out_img_w += border_size
        start_col += border_size
    elif border_type == 'right':
        out_img_w += border_size
    elif border_type == 'top_and_bottom':
        out_img_h += 2 * border_size
        start_row += border_size
    elif border_type == 'left_and_right':
        out_img_w += 2 * border_size
        start_col += border_size

    out_img = np.zeros((out_img_h, out_img_w, 3), dtype=np.uint8)
    out_img[start_row:start_row + img_h, start_col:start_col + img_w, :] = img
    return out_img


def putTextWithBackground(img, text, fmt=None):
    font_types = {
        0: cv2.FONT_HERSHEY_COMPLEX_SMALL,
        1: cv2.FONT_HERSHEY_COMPLEX,
        2: cv2.FONT_HERSHEY_DUPLEX,
        3: cv2.FONT_HERSHEY_PLAIN,
        4: cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        5: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        6: cv2.FONT_HERSHEY_SIMPLEX,
        7: cv2.FONT_HERSHEY_TRIPLEX,
        8: cv2.FONT_ITALIC,
    }
    loc = (5, 15)
    size = 1
    thickness = 1
    col = (255, 255, 255)
    bgr_col = (0, 0, 0)
    font_id = 0

    if fmt is not None:
        try:
            font_id = fmt[0]
            loc = tuple(fmt[1:3])
            size, thickness = fmt[3:5]
            col = tuple(fmt[5:8])
            bgr_col = tuple(fmt[8:])
        except IndexError:
            pass

    disable_bkg = any([k < 0 for k in bgr_col])

    # print('font_id: {}'.format(font_id))
    # print('loc: {}'.format(loc))
    # print('size: {}'.format(size))
    # print('thickness: {}'.format(thickness))
    # print('col: {}'.format(col))
    # print('bgr_col: {}'.format(bgr_col))
    # print('disable_bkg: {}'.format(disable_bkg))

    font = font_types[font_id]

    text_offset_x, text_offset_y = loc
    if not disable_bkg:
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=size, thickness=thickness)[0]
        box_coords = ((text_offset_x, text_offset_y + 5), (text_offset_x + text_width, text_offset_y - text_height))
        cv2.rectangle(img, box_coords[0], box_coords[1], bgr_col, cv2.FILLED)
    cv2.putText(img, text, loc, font, size, col, thickness)


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


# import gmpy
def stackImages(img_list, grid_size=None, stack_order=0, borderless=1,
                preserve_order=0, return_idx=0, annotations=None,
                ann_fmt=(0, 5, 15, 1, 1, 255, 255, 255, 0, 0, 0), only_height=0, sep_size=0):
    for img_id, img in enumerate(img_list):
        if len(img.shape) == 2:
            img_list[img_id] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    n_images = len(img_list)
    # print('grid_size: {}'.format(grid_size))

    if grid_size is None:
        if n_images < 3:
            n_cols, n_rows = n_images, 1
        else:
            n_cols = n_rows = int(np.ceil(np.sqrt(n_images)))

            if n_rows * (n_cols - 1) >= n_images:
                n_cols -= 1
    else:
        n_rows, n_cols = grid_size
    target_ar = 1920.0 / 1080.0
    if n_cols <= n_rows:
        target_ar /= 2.0
    shape_img_id = 0
    min_ar_diff = np.inf
    img_heights = np.zeros((n_images,), dtype=np.int32)
    for _img_id in range(n_images):
        height, width = img_list[_img_id].shape[:2]
        img_heights[_img_id] = height
        img_ar = float(n_cols * width) / float(n_rows * height)
        ar_diff = abs(img_ar - target_ar)
        if ar_diff < min_ar_diff:
            min_ar_diff = ar_diff
            shape_img_id = _img_id

    img_heights_sort_idx = np.argsort(-img_heights)
    row_start_idx = img_heights_sort_idx[:n_rows]
    img_idx = img_heights_sort_idx[n_rows:]
    # print('img_heights: {}'.format(img_heights))
    # print('img_heights_sort_idx: {}'.format(img_heights_sort_idx))
    # print('img_idx: {}'.format(img_idx))

    # grid_size = [n_rows, n_cols]
    img_size = img_list[shape_img_id].shape
    height, width = img_size[:2]

    if only_height:
        width = 0
    # grid_size = [n_rows, n_cols]
    # print 'img_size: ', img_size
    # print 'n_images: ', n_images
    # print 'grid_size: ', grid_size

    # print()
    stacked_img = None
    list_ended = False
    img_idx_id = 0
    inner_axis = 1 - stack_order
    stack_idx = []
    stack_locations = []
    start_row = 0
    curr_ann = ''
    for row_id in range(n_rows):
        start_id = n_cols * row_id
        curr_row = None
        start_col = 0
        for col_id in range(n_cols):
            img_id = start_id + col_id
            if img_id >= n_images:
                curr_img = np.zeros(img_size, dtype=np.uint8)
                list_ended = True
            else:
                if preserve_order:
                    _curr_img_id = img_id
                elif col_id == 0:
                    _curr_img_id = row_start_idx[row_id]
                else:
                    _curr_img_id = img_idx[img_idx_id]
                    img_idx_id += 1

                curr_img = img_list[_curr_img_id]
                if annotations:
                    curr_ann = annotations[_curr_img_id]
                stack_idx.append(_curr_img_id)
                # print(curr_img.shape[:2])

                if curr_ann:
                    putTextWithBackground(curr_img, curr_ann, fmt=ann_fmt)

                if not borderless:
                    curr_img = resizeAR(curr_img, width, height)
                if img_id == n_images - 1:
                    list_ended = True
            if curr_row is None:
                curr_row = curr_img
            else:
                if borderless:
                    curr_img = resizeAR(curr_img, 0, curr_row.shape[0])
                # print('curr_row.shape: ', curr_row.shape)
                # print('curr_img.shape: ', curr_img.shape)

                if sep_size:
                    sep_img_shape = list(curr_row.shape)
                    if inner_axis == 1:
                        sep_img_shape[1] = sep_size
                    else:
                        sep_img_shape[0] = sep_size

                    sep_img = np.full(sep_img_shape, 255, dtype=curr_row.dtype)
                    curr_row = np.concatenate((curr_row, sep_img, curr_img), axis=inner_axis)
                else:
                    curr_row = np.concatenate((curr_row, curr_img), axis=inner_axis)

            curr_h, curr_w = curr_img.shape[:2]
            stack_locations.append((start_row, start_col, start_row + curr_h, start_col + curr_w))
            start_col += curr_w

        if stacked_img is None:
            stacked_img = curr_row
        else:
            if borderless:
                resize_factor = float(curr_row.shape[1]) / float(stacked_img.shape[1])
                curr_row = resizeAR(curr_row, stacked_img.shape[1], 0)
                new_start_col = 0
                for _i in range(n_cols):
                    _start_row, _start_col, _end_row, _end_col = stack_locations[_i - n_cols]
                    _w, _h = _end_col - _start_col, _end_row - _start_row
                    w_resized, h_resized = _w / resize_factor, _h / resize_factor
                    stack_locations[_i - n_cols] = (
                        _start_row, new_start_col, _start_row + h_resized, new_start_col + w_resized)
                    new_start_col += w_resized
            # print('curr_row.shape: ', curr_row.shape)
            # print('stacked_img.shape: ', stacked_img.shape)

            if sep_size:
                sep_img_shape = list(curr_row.shape)
                if stack_order == 1:
                    sep_img_shape[1] = sep_size
                else:
                    sep_img_shape[0] = sep_size
                sep_img = np.full(sep_img_shape, 255, dtype=curr_row.dtype)
                stacked_img = np.concatenate((stacked_img, sep_img, curr_row), axis=stack_order)
            else:
                stacked_img = np.concatenate((stacked_img, curr_row), axis=stack_order)

        curr_h, curr_w = curr_row.shape[:2]
        start_row += curr_h

        if list_ended:
            break
    if return_idx:
        return stacked_img, stack_idx, stack_locations
    else:
        return stacked_img
