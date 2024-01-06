import os
import paramparse


class Params:
    def __init__(self):
        self.in_path = '.'
        self.in_ext = '.html'

        self.recursive = 1


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def process(in_txt):
    lines = in_txt.split('\n')
    lines = [line for line in lines if line.strip()]

    start_token = 'href="magnet:?'
    end_token = '2Fannounce"'

    for line in lines:
        if start_token not in line:
            continue
        start_idx = line.find(start_token)
        end_idx = line.find(end_token)

        magnet_txt = line[start_idx + 6:end_idx + len(end_token) - 1]

        magnet_txt = magnet_txt.replace("&amp;", "&")

        return magnet_txt

    return None


def copy_from_clipboard():
    try:
        import win32clipboard

        win32clipboard.OpenClipboard()
        in_txt = win32clipboard.GetClipboardData()
    except BaseException as e:
        print('GetClipboardData failed: {}'.format(e))
        win32clipboard.CloseClipboard()
        return None
    win32clipboard.CloseClipboard()
    return in_txt


def copy_to_clipboard(out_txt, print_txt=0):
    if print_txt:
        print(out_txt)
    try:
        # win32clipboard.OpenClipboard()
        # win32clipboard.SetClipboardText(out_txt)
        import pyperclip

        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))
        return

    # win32clipboard.CloseClipboard()


def main():
    params = Params()
    paramparse.process(params)

    in_txt = copy_from_clipboard()

    assert os.path.isdir(params.in_path), "invalid mht path: {}".format(params.in_path)

    if params.recursive:
        files_gen = [[linux_path(dirpath, f) for f in filenames if
                      f.endswith(params.in_ext)]
                     for (dirpath, dirnames, filenames) in os.walk(params.in_path, followlinks=True)]
        files = [item for sublist in files_gen for item in sublist]
    else:
        files = os.listdir(params.in_path)
        files = [linux_path(params.in_path, k) for k in files if k and k.endswith(params.in_ext)]

    files.sort(key=os.path.getmtime)
    n_files = len(files)

    magnet_links = []
    for file_id, file in enumerate(files):
        print('reading file {} / {}: {}'.format(file_id + 1, n_files, file))

        # file = dst_file

        in_txt = open(file, 'r', encoding="utf8").read()
        magnet_link = process(in_txt)

        if magnet_link is None:
            print(f'\nno magnet link found in {file}\n')
            continue

        magnet_links.append(magnet_link)

    magnet_links_str = '\n\n'.join(magnet_links)

    out_file = linux_path(params.in_path, "magnet_links.txt")
    with open(out_file, 'w') as fid:
        fid.write(magnet_links_str)


if __name__ == '__main__':
    main()
