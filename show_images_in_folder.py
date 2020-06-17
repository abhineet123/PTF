import cv2
import sys
import time
import ctypes
import platform
import os, shutil
from datetime import datetime
from pprint import pformat

from Misc import processArguments, sortKey, resizeAR


def clear_dir(dirname):
    for filename in os.listdir(dirname):
        file_path = os.path.join(dirname, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_free_space_mb(dirname):
    """Return folder/drive free space (in megabytes)."""
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(dirname), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value / 1024 / 1024
    else:
        st = os.statvfs(dirname)
        return st.f_bavail * st.f_frsize / 1024 / 1024


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
        'codec': 'H264',
        'ext': 'mkv',
        'out_postfix': '',
        'reverse': 0,
        'min_free_space': 30,
    }

    processArguments(sys.argv[1:], params)
    src_path = params['src_path']
    min_free_space = params['min_free_space']

    img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff', '.webp')

    # existing_images = {}
    image_pause = {}
    _pause = 1

    src_path = os.path.abspath(src_path)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(script_dir, 'siif_log.txt')
    with open(log_path, 'w') as fid:
        fid.write(src_path)

    os.environ["SIIF_DUMP_IMAGE_PATH"] = src_path

    read_img_path = os.path.join(src_path, "read")

    if os.path.exists(read_img_path):
        clear_dir(read_img_path)
        # shutil.rmtree(read_img_path)
    else:
        os.makedirs(read_img_path)

    print('SIIF started in {}'.format(src_path))

    img_id = 0
    exit_program = 0
    while not exit_program:
        _src_files = [k for k in os.listdir(src_path) if
                      os.path.splitext(k.lower())[1] in img_exts]
        for _src_file in _src_files:
            _src_path = os.path.join(src_path, _src_file)
            # mod_time = os.path.getmtime(_src_path)
            _src_file_no_ext = os.path.splitext(_src_file)[0]
            try:
                _src_file_id, _src_file_timestamp = _src_file_no_ext.split('___')
            except ValueError as e:
                print('Invalid _src_file: {} with _src_file_no_ext: {}'.format(
                    _src_file, _src_file_no_ext))

            # if _src_file_id not in existing_images or existing_images[_src_file_id] != _src_file_timestamp:
            #     existing_images[_src_file] = _src_file_timestamp
            print('reading {} with time: {}'.format(_src_file_id, _src_file_timestamp))

            img = cv2.imread(_src_path)

            # out_path = _src_path+'.done'
            # print('writing to {}'.format(out_path))
            # with open(out_path, 'w') as fid:
            #     fid.write('siif')
            #     fid.close()

            _dst_path = os.path.join(read_img_path, os.path.basename(_src_path))

            # os.remove(_src_path)
            shutil.move(_src_path, _dst_path)

            cv2.imshow(_src_file_id, img)

            if _src_file_id not in image_pause:
                image_pause[_src_file_id] = _pause

            k = cv2.waitKey(1 - image_pause[_src_file_id])
            if k == 27:
                exit_program = 1
                break
            elif k == 32:
                _pause = 1 - _pause
                for _src_file_id in image_pause:
                    image_pause[_src_file_id] = _pause
            elif k == ord('q'):
                cv2.destroyWindow(_src_file_id)
                del image_pause[_src_file_id]
            elif k == ord('p'):
                image_pause[_src_file_id] = 1 - image_pause[_src_file_id]

            img_id += 1

            print('image_pause: {}'.format(image_pause))

            if img_id % 100 == 0:
                free_space = get_free_space_mb(src_path)
                print('free_space {}'.format(free_space))
                if free_space < min_free_space:
                    print('Free space running low. Press any key to clear the backup directory')
                    cv2.waitKey(0)
                    clear_dir(read_img_path)

            # del_images = []
            # for existing_image in existing_images.keys():
            #     if existing_image not in _src_files:
            #         cv2.destroyWindow(existing_image)
            #         del_images.append(existing_image)
            #
            # for del_image in del_images:
            #     del existing_images[del_image]

        cv2.waitKey(1)

    for _src_file_id in image_pause.keys():
        cv2.destroyWindow(_src_file_id)


if __name__ == '__main__':
    main()
