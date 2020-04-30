import cv2
import sys
import functools
import os
import time
import shutil
from skimage import measure
from skimage import feature

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

from Misc import processArguments, sortKey, resizeAR, col_rgb


def getPlotImage(data_x, data_y, cols, title, line_labels, x_label, y_label,
                 scatter=None, ylim=None, legend=0):
    cols = [(col[0] / 255.0, col[1] / 255.0, col[2] / 255.0) for col in cols]

    fig = Figure(
        # figsize=(6.4, 3.6), dpi=300,
        figsize=(4.8, 2.7), dpi=400,
        # edgecolor='k',
        # facecolor ='k'
    )
    # fig.tight_layout()
    # fig.set_tight_layout(True)
    fig.subplots_adjust(
        bottom=0.17,
        right=0.95,
    )
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    n_data = len(data_y)
    for i in range(n_data):
        datum_y = data_y[i]
        line_label = line_labels[i]
        col = cols[i]
        args = {
            'color': col
        }
        if legend:
            args['label'] = line_label

        data_x = np.asarray(data_x).squeeze()
        datum_y = np.asarray(datum_y).squeeze()

        ax.plot(data_x, datum_y, **args)

    if scatter is not None:
        ax.scatter(scatter, datum_y[scatter], s=20, c='r')

    plt.rcParams['axes.titlesize'] = 10
    # fontdict = {'fontsize': plt.rcParams['axes.titlesize'],
    #             'fontweight': plt.rcParams['axes.titleweight'],
    # 'verticalalignment': 'baseline',
    # 'horizontalalignment': plt.loc
    # }
    ax.set_title(title,
                 # fontdict=fontdict
                 )
    if legend:
        ax.legend(fancybox=True, framealpha=0.1)
    ax.grid(1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if ylim is not None:
        ax.set_ylim(*ylim)

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    plot_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(
        int(height), int(width), 3)

    return plot_img


def optical_flow_lk_fb(prev_gray, gray):
    feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    prev_fb, status, error = cv2.calcOpticalFlowPyrLK(gray, prev_gray, next, None, **lk_params)

    fb_error = np.linalg.norm((next - prev_fb).flatten())

    return fb_error


def optical_flow_farneback_fb(prev_gray, gray):
    flow_fw = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_bw = cv2.calcOpticalFlowFarneback(gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    fb_error = np.linalg.norm((flow_fw + flow_bw).flatten())
    return fb_error


def main():
    params = {
        'src_path': '.',
        'save_path': '',
        'img_ext': 'jpg',
        'show_img': 1,
        'del_src': 0,
        'start_id': 0,
        'width': 0,
        'height': 0,
        'fps': 30,
        # 'codec': 'FFV1',
        # 'ext': 'avi',
        'codec': 'H264',
        'ext': 'mkv',
        'out_postfix': '',
        'labels_col': 'red',
        'reverse': 0,
        'sub_seq_start_id': 0,
        'metric': 4,
        'thresh': -1,
        'order': 5,
        'frames_per_seq': 0,
        'video_mode': 0,
    }

    processArguments(sys.argv[1:], params)
    _src_path = params['src_path']
    show_img = params['show_img']
    _start_id = params['start_id']
    _width = params['width']
    _height = params['height']
    reverse = params['reverse']
    labels_col = params['labels_col']
    metric = params['metric']
    _thresh = params['thresh']
    order = params['order']
    sub_seq_start_id = params['sub_seq_start_id']
    frames_per_seq = params['frames_per_seq']
    video_mode = params['video_mode']
    codec = params['codec']

    vid_exts = ['.mkv', '.mp4', '.avi', '.mjpg', '.wmv', '.gif', '.webm']
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

    min_thresh = 0
    if metric == 0:
        sim_func = measure.compare_mse
        metric_type = 'MSE'
        cmp_func = np.greater
        max_thresh = 10000
    elif metric == 1:
        sim_func = measure.compare_ssim
        metric_type = 'SSIM'
        cmp_func = np.less
        max_thresh = 1
    elif metric == 2:
        sim_func = measure.compare_nrmse
        metric_type = 'NRMSE'
        cmp_func = np.greater
    elif metric == 3:
        sim_func = measure.compare_psnr
        metric_type = 'PSNR'
        cmp_func = np.less
    elif metric == 4:
        sim_func = functools.partial(cv2.matchTemplate, method=cv2.TM_CCORR_NORMED)
        metric_type = 'NCC'
        cmp_func = np.less
        max_thresh = 1
    elif metric == 5:
        sim_func = optical_flow_lk_fb
        metric_type = 'LK'
        cmp_func = np.greater
    elif metric == 6:
        sim_func = optical_flow_farneback_fb
        metric_type = 'Farneback'
        cmp_func = np.greater

    metric_type_ratio = f'{metric_type} Ratio'

    _src_path = os.path.abspath(_src_path)

    vid_ext = None

    if any(_src_path.endswith(_ext) for _ext in vid_exts):
        vid_fname, vid_ext = os.path.splitext(os.path.basename(_src_path))
        if not video_mode:
            print('Converting video to image sequences: {}'.format(_src_path))
            os.system('v2i {}'.format(_src_path))

            _src_path = os.path.join(os.path.dirname(_src_path), vid_fname)

    if os.path.isdir(_src_path):
        if video_mode:
            src_paths = [k for k in os.listdir(_src_path) for _ext in vid_exts if k.endswith(_ext)]
        else:
            src_files = [k for k in os.listdir(_src_path) for _ext in img_exts if k.endswith(_ext)]
            if not src_files:
                src_paths = [os.path.join(_src_path, k) for k in os.listdir(_src_path) if
                             os.path.isdir(os.path.join(_src_path, k))]
            else:
                src_paths = [_src_path]
    elif os.path.isfile(_src_path):
        if video_mode:
            src_paths = [_src_path]
        else:
            print('Reading source image sequences from: {}'.format(_src_path))
            src_paths = [x.strip() for x in open(_src_path).readlines() if x.strip()]
        n_seq = len(src_paths)
        if n_seq <= 0:
            raise SystemError('No input sequences found')
        print('n_seq: {}'.format(n_seq))
    else:
        raise IOError('Invalid src_path: {}'.format(_src_path))

    if reverse == 1:
        print('Writing the reverse sequence')
    elif reverse == 2:
        print('Appending the reverse sequence')

    labels_col_rgb = col_rgb[labels_col]
    plot_cols = [labels_col_rgb, ]

    for src_path in src_paths:

        src_path = os.path.abspath(src_path)

        start_id = _start_id
        thresh = _thresh
        seq_name = os.path.basename(src_path)
        print('Reading source images from: {}'.format(src_path))

        if not video_mode:

            src_files = [k for k in os.listdir(src_path) for _ext in img_exts if k.endswith(_ext)]
            n_src_files = len(src_files)
            if n_src_files <= 0:
                raise SystemError('No input frames found')

            # print('src_files: {}'.format(src_files))

            src_files.sort(key=sortKey)
            print('n_src_files: {}'.format(n_src_files))

            if reverse == 1:
                src_files = src_files[::-1]
            elif reverse == 2:
                src_files += src_files[::-1]
                n_src_files *= 2

            filename = src_files[start_id]
            file_path = os.path.join(src_path, filename)
            assert os.path.exists(file_path), f'Image file {file_path} does not exist'
            prev_image = cv2.imread(file_path)
        else:
            src_files = None
            cap = cv2.VideoCapture(src_path)
            n_src_files = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_id)
            ret, prev_image = cap.read()
            if not ret:
                raise IOError('frame {} could not be read'.format(start_id))

        prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

        frame_id = start_id + 1
        pause_after_frame = 0

        if frames_per_seq > 0:
            split_indices = list(range(frames_per_seq, n_src_files, frames_per_seq))
        else:
            sim_list = []
            sim_ratio_list = []
            prev_sim = None
            # sub_seq_id = sub_seq_start_id
            # if thresh >= 0:
            #     dst_path = os.path.join(src_path, f'{sub_seq_id}')
            #     if not os.path.isdir(dst_path):
            #         os.makedirs(dst_path)
            print_diff = max(1, int(n_src_files / 100))
            start_t = time.time()
            while True:
                if not video_mode:
                    filename = src_files[frame_id]
                    file_path = os.path.join(src_path, filename)
                    assert os.path.exists(file_path), f'Image file {file_path} does not exist'
                    image = cv2.imread(file_path)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_id)
                    ret, image = cap.read()
                    if not ret:
                        raise IOError('frame {} could not be read'.format(start_id))

                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                if image.shape != prev_image.shape:
                    sim = min_thresh
                else:
                    sim = sim_func(image, prev_image)

                if prev_sim is not None:
                    s_ratio = (sim + 1) / (prev_sim + 1)
                else:
                    s_ratio = 1

                sim_list.append(sim)
                sim_ratio_list.append(s_ratio)

                prev_image = image
                prev_sim = sim

                # image = resizeAR(image, width, height)

                if show_img:
                    sim_plot = getPlotImage([list(range(start_id, frame_id)), ], [sim_list, ], plot_cols,
                                            metric_type, [metric_type, ], 'frame', metric_type)

                    sim_ratio_plot = getPlotImage([list(range(start_id, frame_id)), ], [sim_ratio_list, ], plot_cols,
                                                  metric_type_ratio, [metric_type_ratio, ], 'frame', metric_type_ratio)

                    cv2.imshow('sim_plot', sim_plot)
                    cv2.imshow('sim_ratio_plot', sim_ratio_plot)
                    cv2.imshow(seq_name, image)
                    k = cv2.waitKey(1 - pause_after_frame) & 0xFF
                    if k == ord('q') or k == 27:
                        break
                    elif k == 32:
                        pause_after_frame = 1 - pause_after_frame

                # if thresh >= 0:
                #     if cmp_func(sim, thresh):
                #         sub_seq_id += 1
                #         print(f'sub_seq_id: {sub_seq_id} with sim: {sim}')
                #         dst_path = os.path.join(src_path, f'{sub_seq_id}')
                #         if not os.path.isdir(dst_path):
                #             os.makedirs(dst_path)
                #     dst_file_path = os.path.join(dst_path, filename)
                #     shutil.move(file_path, dst_file_path)

                frame_id += 1

                if frame_id % print_diff == 0:
                    end_t = time.time()
                    fps = float(print_diff) / (end_t - start_t)
                    sys.stdout.write('\rDone {:d}/{:d} frames at {:.4f} fps'.format(
                        frame_id - start_id, n_src_files - start_id, fps))
                    sys.stdout.flush()
                    start_t = end_t

                if frame_id >= n_src_files:
                    break

            sys.stdout.write('\n\n')
            sys.stdout.flush()

            """
            compensate for the 1-frame differential
            """
            sim_list.insert(0, sim_list[0])
            sim_ratio_list.insert(0, sim_ratio_list[0])

            sim_list = np.asarray(sim_list).squeeze()
            sim_ratio_list = np.asarray(sim_ratio_list).squeeze()

            split_indices = []

            if thresh < 0:
                if thresh == -1:
                    _data = sim_list
                    # c_max_index = argrelextrema(sim_list, cmp_func, order=order)
                    # plt.plot(sim_list)
                    # plt.scatter(c_max_index[0], sim_list[c_max_index[0]], linewidth=0.3, s=50, c='r')
                    # plt.show()
                elif thresh == -2:
                    _data = sim_ratio_list
                    # plt.plot(sim_ratio_list)
                    # plt.scatter(c_max_index[0], sim_ratio_list[c_max_index[0]], linewidth=0.3, s=50, c='r')
                    # plt.show()

                def update_order(_order):
                    nonlocal order, split_indices
                    order = _order
                    split_indices = argrelextrema(_data, cmp_func, order=order)[0]
                    split_indices = [k for k in split_indices if cmp_func(_data[k], thresh)]
                    scatter_plot = getPlotImage([list(range(len(_data))), ], [_data, ], plot_cols,
                                                metric_type, [metric_type, ], 'frame', metric_type,
                                                scatter=split_indices)
                    print(f'order: {order}')
                    cv2.imshow('scatter_plot', scatter_plot)

                def update_thresh(x):
                    nonlocal thresh, split_indices
                    thresh = min_thresh + float(max_thresh - min_thresh) / float(1000) * x
                    split_indices = argrelextrema(_data, cmp_func, order=order)[0]
                    split_indices = [k for k in split_indices if cmp_func(_data[k], thresh)]
                    scatter_plot = getPlotImage([list(range(len(_data))), ], [_data, ], plot_cols,
                                                metric_type, [metric_type, ], 'frame', metric_type,
                                                scatter=split_indices)
                    print(f'thresh: {thresh}')
                    cv2.imshow('scatter_plot', scatter_plot)

                update_order(order)
                cv2.createTrackbar('order', 'scatter_plot', order, 100, update_order)
                cv2.createTrackbar('threshold', 'scatter_plot', 0, 1000, update_thresh)

            else:
                if thresh == 0:
                    _data = sim_list
                else:
                    _data = sim_ratio_list
                max_thresh = np.max(_data)

                def update_thresh(x):
                    nonlocal thresh, split_indices
                    if x == 0:
                        return
                    thresh = min_thresh + float(max_thresh) / float(x)
                    split_indices = np.nonzero(cmp_func(_data, thresh))
                    scatter_plot = getPlotImage([list(range(len(_data))), ], [_data, ], plot_cols,
                                                metric_type, [metric_type, ], 'frame', metric_type,
                                                scatter=split_indices)
                    cv2.imshow('scatter_plot', scatter_plot)

                update_order(order)
                cv2.createTrackbar('threshold', 'scatter_plot', 0, 1000, update_thresh)

            while True:
                k = cv2.waitKey(0)
                print('k: {}'.format(k))
                if k == 13 or k == 27:
                    break

            cv2.destroyWindow('scatter_plot')

            if show_img:
                cv2.destroyAllWindows()

            if k == 27:
                break

            print(f'order: {order}')
            print(f'thresh: {thresh}')
            print(f'n_src_files: {n_src_files}')

        if n_src_files not in split_indices:
            split_indices.append(n_src_files)

        split_indices = list(split_indices)
        n_splits = len(split_indices)
        print(f'Splitting into {n_splits} sub sequences:\n{split_indices}')
        start_id = 0
        sub_seq_id = sub_seq_start_id
        for end_id in split_indices:
            print(f'sub_seq_id: {sub_seq_id} with start_id: {start_id}, end_id: {end_id}')

            if video_mode:
                w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

                dst_path = os.path.join(src_path, f'{seq_name}_{sub_seq_id}.{vid_ext}')
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_out = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
                if video_out is None:
                    raise IOError('Output video file could not be opened: {}'.format(dst_path))

                for i in range(start_id, end_id):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_id)
                    ret, image = cap.read()
                    if not ret:
                        raise IOError('frame {} could not be read'.format(start_id))
                    video_out.write(image)
            else:
                dst_path = os.path.join(src_path, f'{seq_name}_{sub_seq_id}')
                if not os.path.isdir(dst_path):
                    os.makedirs(dst_path)
                for i in range(start_id, end_id):
                    filename = src_files[i]
                    src_file_path = os.path.join(src_path, filename)
                    dst_file_path = os.path.join(dst_path, filename)
                    shutil.move(src_file_path, dst_file_path)

            start_id = end_id
            sub_seq_id += 1


if __name__ == '__main__':
    main()
