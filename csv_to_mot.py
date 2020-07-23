import os
import sys
import glob
import pandas as pd

from Misc import processArguments, sortKey

_params = {
    'root_dir': '.',
    'img_root_dir': '/data/DETRAC/Images',
    'start_id': 0,
    'end_id': -1,
    'ignored_region_only': 0,
    'speed': 0.5,
    'show_img': 0,
    'quality': 3,
    'resize': 0,
    'mode': 0,
    'auto_progress': 0,
}

processArguments(sys.argv[1:], _params)

root_dir = _params['root_dir']
img_root_dir = _params['img_root_dir']
start_id = _params['start_id']
ignored_region_only = _params['ignored_region_only']
end_id = _params['end_id']

csv_exts = ('.csv', )
img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff', '.webp')

__csv_files_list = [os.path.join(root_dir, k) for k in os.listdir(root_dir) if
                      os.path.splitext(k.lower())[1] in csv_exts]

if end_id <= start_id:
    end_id = len(__csv_files_list) - 1

print('root_dir: {}'.format(root_dir))
print('start_id: {}'.format(start_id))
print('end_id: {}'.format(end_id))


print('__csv_files_list: {}'.format(__csv_files_list))

n_frames_list = []

for seq_id in range(start_id, end_id + 1):
    csv_path = __csv_files_list[seq_id]
    seq_name = os.path.splitext(os.path.basename(csv_path))[0]

    img_path = os.path.join(img_root_dir, seq_name)

    _src_files = [k for k in os.listdir(img_path) if
                  os.path.splitext(k.lower())[1] in img_exts]
    _src_files.sort(key=sortKey)

    bounding_boxes = []

    print('Processing sequence {:d} :: {:s}'.format(seq_id, seq_name))
    print('Reading from {}'.format(csv_path))
    mot_path = csv_path.replace('.csv', '.txt')
    out_fid = open(mot_path, 'w')
    print('Writing to {}'.format(mot_path))

    df_det = pd.read_csv(csv_path)

    for _, row in df_det.iterrows():
        filename = row['filename']

        confidence = row['confidence']
        xmin = float(row['xmin'])
        ymin = float(row['ymin'])
        xmax = float(row['xmax'])
        ymax = float(row['ymax'])
        # width = float(row['width'])
        # height = float(row['height'])
        class_name = row['class']

        w, h = xmax - xmin, ymax - ymin

        try:
            frame_id = _src_files.index(filename)
        except:
            raise IOError('Invalid filename found: {}'.format(filename))

        out_fid.write('{:d},{:d},{:f},{:f},{:f},{:f},{:f},-1,-1,-1\n'.format(
            frame_id + 1, -1, xmin, ymin, w, h, confidence))


        # bounding_boxes.append(
        #     {"class": class_name,
        #      "confidence": confidence,
        #      "filename": filename,
        #      # "width": width,
        #      # "height": height,
        #      "bbox": [xmin, ymin, xmax, ymax]}
        # )

    # for bndbox in bounding_boxes:
    #
    #     xmin, ymin, xmax, ymax = bndbox['bbox']


    out_fid.close()

