import cv2
import time
import paramparse
import os, shutil
from datetime import datetime
import pandas as pd
import numpy as np

"""BGR values for different colors"""
col_bgr = {
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


def show_filtered_detections(img, all_detections, thresh, show_all_classes, win_name):
    valid_detections = [k for k in all_detections if k[5] >= thresh]
    if not show_all_classes:
        valid_detections = [k for k in valid_detections if k[4] == 1]

    img_disp = np.copy(img)

    draw_detections(img_disp, valid_detections, _id=None, color='black', thickness=1,
                    is_dotted=0, transparency=0.)
    cv2.imshow(win_name, img_disp)
    return valid_detections


def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=7):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    for p in pts:
        cv2.circle(img, p, thickness, color, -1)


def draw_dotted_poly(img, pts, color, thickness=1):
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        draw_dotted_line(img, s, e, color, thickness)


def draw_dotted_rect(img, pt1, pt2, color, thickness=1):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    draw_dotted_poly(img, pts, color, thickness)


def draw_detections(frame, boxes, _id=None, color='black', thickness=2,
                    is_dotted=0, transparency=0.):
    boxes = np.asarray(boxes).reshape((-1, 4))

    for box in boxes:
        draw_box(frame, box, _id, color, thickness,
                 is_dotted, transparency)


def draw_box(frame, box, _id=None, color='black', thickness=2,
             is_dotted=0, transparency=0., text_col=None):
    """
    :type frame: np.ndarray
    :type box: np.ndarray
    :type _id: int | str | None
    :param color: indexes into col_rgb
    :type color: str
    :type thickness: int
    :type is_dotted: int
    :type transparency: float
    :rtype: None
    """
    box = np.asarray(box)

    if np.any(np.isnan(box)):
        print('invalid location provided: {}'.format(box))
        return

    box = box.squeeze()
    pt1 = (int(box[0]), int(box[1]))
    pt2 = (int(box[2]), int(box[3]))

    if transparency > 0:
        _frame = np.copy(frame)
    else:
        _frame = frame

    if is_dotted:
        draw_dotted_rect(_frame, pt1, pt2, col_bgr[color], thickness=thickness)
    else:
        cv2.rectangle(_frame, pt1, pt2, col_bgr[color], thickness=thickness)

    if transparency > 0:
        frame[pt1[1]:pt2[1], pt1[0]:pt2[0], ...] = (
                frame[pt1[1]:pt2[1], pt1[0]:pt2[0], ...].astype(np.float32) * (1 - transparency) +
                _frame[pt1[1]:pt2[1], pt1[0]:pt2[0], ...].astype(np.float32) * transparency
        ).astype(frame.dtype)

    if _id is not None:
        if text_col is None:
            text_col = color
        text_col = col_bgr[text_col]

        font_line_type = cv2.LINE_AA

        cv2.putText(frame, str(_id), (int(box[0] - 1), int(box[1] - 10)),
                    # cv2.FONT_HERSHEY_SIMPLEX,
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.50, text_col, 1, font_line_type)


def main():
    params = {
        'src_path': '.',
        'save_path': '',
        'save_root_dir': '',
        'img_ext': 'jpg',
        'show_img': 1,
        'del_src': 0,
        'start_id': 0,
        'n_frames': 0,
        'width': 0,
        'height': 0,
        'fps': 30,
        # 'codec': 'FFV1',
        # 'ext': 'avi',
        'sleep': 0.,
        'codec': 'H264',
        'ext': 'mkv',
        'csv_ext': 'txt',
        'out_postfix': '',
        'reverse': 0,
        'min_free_space': 30,
    }

    paramparse.process_dict(params)
    src_path = params['src_path']
    min_free_space = params['min_free_space']
    sleep = params['sleep']
    csv_ext = params['csv_ext']

    img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff', '.webp')

    src_path = os.path.abspath(src_path)
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # log_path = os.path.join(script_dir, 'siif_log.txt')
    # with open(log_path, 'w') as fid:
    #     fid.write(src_path)
    #
    # os.environ["MIIF_DUMP_IMAGE_PATH"] = src_path

    read_img_path = os.path.join(src_path, "read")

    if not os.path.exists(read_img_path):
        os.makedirs(read_img_path)

    print('SDIF started in {}'.format(src_path))

    max_thresh = 100
    min_thresh = 0
    thresh = 0
    show_all_classes = 0
    win_name = 'detections'
    img = None

    def update_thresh(x):
        nonlocal thresh
        thresh = float(x) / 100.
        show_filtered_detections(img, all_detections, thresh, show_all_classes, win_name)

    def update_show_all_classes(x):
        nonlocal show_all_classes
        show_all_classes = int(x)
        show_filtered_detections(img, all_detections, thresh, show_all_classes, win_name)

    cv2.createTrackbar('threshold', win_name, int(thresh*100), 100, update_thresh)
    cv2.createTrackbar('show_all_classes', win_name, show_all_classes, 2, update_show_all_classes)

    exit_program = 0
    while not exit_program:
        if sleep > 0:
            time.sleep(sleep)

        _src_files = [k for k in os.listdir(src_path) if
                      os.path.splitext(k.lower())[1] in img_exts]

        for _src_file in _src_files:
            img_src_path = os.path.join(src_path, _src_file)
            _src_file_no_ext, _src_file_ext = os.path.splitext(_src_file)
            time_stamp = datetime.now().strftime("_%y%m%d_%H%M%S_%f")

            img = cv2.imread(img_src_path)
            height, width = img.shape[:2]

            csv_src_path = os.path.join(src_path, '{}.{}'.format(_src_file_no_ext, csv_ext))
            df = pd.read_csv(csv_src_path)

            n_detections = len(df.index)
            all_detections = []
            for i in range(n_detections):
                df_bbox = df.iloc[i]
                xmin = df_bbox.loc['x1']
                ymin = df_bbox.loc['y1']
                xmax = df_bbox.loc['x2']
                ymax = df_bbox.loc['y2']
                class_id = df_bbox.loc['class_id']
                score = df_bbox.loc['score']

                all_detections.append([xmin, ymin, xmax, ymax, class_id, score])

            show_filtered_detections(img, all_detections, thresh, show_all_classes, win_name)

            w = xmax - xmin
            h = ymax - ymin

            if w < 0 or h < 0:
                print('\nInvalid box in image {} with dimensions {} x {}\n'.format(_src_file, w, h))
                xmin, xmax = xmax, xmin
                ymin, ymax = ymax, ymin
                w = xmax - xmin
                h = ymax - ymin
                if w < 0 or h < 0:
                    raise IOError('\nSwapping corners still giving invalid dimensions {} x {}\n'.format(w, h))

            # if w < min_size or h < min_size:
            #     print('\nIgnoring image {} with too small {} x {} box '.format(image_name, w, h))
            #     return None, None, None
            #     # continue

            def clamp(x, min_value=0.0, max_value=1.0):
                return max(min(x, max_value), min_value)

            xmin = clamp(xmin / width)
            xmax = clamp(xmax / width)
            ymin = clamp(ymin / height)
            ymax = clamp(ymax / height)
            #
            # xmins.append(xmin)
            # xmaxs.append(xmax)
            # ymins.append(ymin)
            # ymaxs.append(ymax)
            #
            # classes_text.append(class_name)
            # class_ids.append(class_id)

        img_dst_path = os.path.join(read_img_path, _src_file_no_ext + time_stamp + _src_file_ext)
        csv_dst_path = os.path.join(read_img_path, _src_file_no_ext + time_stamp + '.' + csv_ext)
        print(f'{img_src_path} -> {img_dst_path}')

        shutil.move(img_src_path, img_dst_path)
        shutil.move(csv_src_path, csv_dst_path)


if __name__ == '__main__':
    main()
