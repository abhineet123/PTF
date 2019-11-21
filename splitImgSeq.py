import cv2
import sys
import os, shutil
from skimage import measure

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from Misc import processArguments, sortKey, resizeAR, col_rgb



def getPlotImage(data_x, data_y, cols, title, line_labels, x_label, y_label, ylim=None, legend=1):
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
    plot_img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(
        int(height), int(width), 3)

    return plot_img

def main():
    params = {
        'src_path': '.',
        'save_path': '',
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
        'codec': 'H264',
        'ext': 'mkv',
        'out_postfix': '',
        'labels_col': 'red',
        'reverse': 0,
    }

    processArguments(sys.argv[1:], params)
    _src_path = params['src_path']
    save_path = params['save_path']
    img_ext = params['img_ext']
    show_img = params['show_img']
    del_src = params['del_src']
    start_id = params['start_id']
    n_frames = params['n_frames']
    _width = params['width']
    _height = params['height']
    fps = params['fps']
    codec = params['codec']
    ext = params['ext']
    out_postfix = params['out_postfix']
    reverse = params['reverse']
    labels_col = params['labels_col']
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']



    if os.path.isdir(_src_path):
        src_files = [k for k in os.listdir(_src_path) for _ext in img_exts if k.endswith(_ext)]
        if not src_files:
            src_paths = [os.path.join(_src_path, k) for k in os.listdir(_src_path) if
                         os.path.isdir(os.path.join(_src_path, k))]
        else:
            src_paths = [_src_path]
    elif os.path.isfile(_src_path):
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
        seq_name = os.path.basename(src_path)

        print('Reading source images from: {}'.format(src_path))

        src_path = os.path.abspath(src_path)
        src_files = [k for k in os.listdir(src_path) for _ext in img_exts if k.endswith(_ext)]
        n_src_files = len(src_files)
        if n_src_files <= 0:
            raise SystemError('No input frames found')
        src_files.sort(key=sortKey)
        print('n_src_files: {}'.format(n_src_files))

        if reverse == 1:
            src_files = src_files[::-1]
        elif reverse == 2:
            src_files += src_files[::-1]
            n_src_files *= 2

        width, height = _width, _height

        if not save_path:
            save_fname = '{}_{}'.format(os.path.basename(src_path), fps)

            if height > 0 and width > 0:
                save_fname = '{}_{}x{}'.format(save_fname, width, height)

            if out_postfix:
                save_fname = '{}_{}'.format(save_fname, out_postfix)

            if reverse:
                save_fname = '{}_r{}'.format(save_fname, reverse)

            save_path = os.path.join(os.path.dirname(src_path), '{}.{}'.format(save_fname, ext))

        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if height <= 0 or width <= 0:
            temp_img = cv2.imread(os.path.join(src_path, src_files[0]))
            height, width, _ = temp_img.shape

        filename = src_files[start_id]
        file_path = os.path.join(src_path, filename)
        assert os.path.exists(file_path), f'Image file {file_path} does not exist'
        prev_image = cv2.imread(file_path)
        prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

        frame_id = start_id + 1
        pause_after_frame = 0

        sim_list = []
        while True:
            filename = src_files[frame_id]
            file_path = os.path.join(src_path, filename)
            assert os.path.exists(file_path), f'Image file {file_path} does not exist'

            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            s = measure.compare_ssim(image, prev_image)
            sim_list.append(s)

            prev_image = image

            # image = resizeAR(image, width, height)

            if show_img:
                sim_plot = getPlotImage([list(range(start_id, frame_id)),], [sim_list, ], plot_cols, 'SSIM', 'SSIM',
                                       'frame', 'SSIM')
                cv2.imshow('sim_plot', sim_plot)
                cv2.imshow(seq_name, image)
                k = cv2.waitKey(1 - pause_after_frame) & 0xFF
                if k == ord('q') or k == 27:
                    break
                elif k == 32:
                    pause_after_frame = 1 - pause_after_frame


            frame_id += 1
            sys.stdout.write('\rDone {:d} frames '.format(frame_id - start_id))
            sys.stdout.flush()

            if n_frames > 0 and (frame_id - start_id) >= n_frames:
                break

            if frame_id >= n_src_files:
                break

        sys.stdout.write('\n\n')
        sys.stdout.flush()

        if show_img:
            cv2.destroyWindow(seq_name)

        if del_src:
            print('Removing source folder {}'.format(src_path))
            shutil.rmtree(src_path)

        save_path = ''

if __name__ == '__main__':
    main()