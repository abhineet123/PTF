import os
import difflib
import shutil
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone

import paramparse


class Params:
    def __init__(self):
        self.src = ''
        self.dst = ''
        self.exclude_links = 1
        self.extra_info = 1
        self.verbose = 0


# def is_dir(path, verbose):
#     try:
#         if path.is_dir():
#             return True
#     except OSError as e:
#         if verbose:
#             print(f'skipping {str(path)}: {e}')
#         return None
#     except BaseException as e:
#         print(f'file access error :: {str(path)}: {e}')
#         input('press any key to exit')
#         return None
#     return False


def tree_to_file(src, dst, exclude_links, extra_info, verbose):
    src = Path(src)
    # prefix components:
    space = '    '
    branch = '│   '
    # pointers:
    tee = '├── '
    last = '└── '

    def recursive_tree(dir_path: Path, prefix: str = ''):
        """A recursive generator, given a directory Path object
        will yield a visual tree structure line by line
        with each line prefixed by the same characters
        """
        try:
            contents = list(dir_path.iterdir())
        except PermissionError as e:
            print(f'\ndir access error: {str(dir_path)}: {e}')
            return

        # contents each get pointers that are ├── with a final └── :
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            name = path.name

            if extra_info:
                try:
                    is_file = path.is_file()
                except OSError as e:
                    pass
                except BaseException as e:
                    print(f'file access error :: {str(path)}: {e}')
                    input('press any key to exit')
                    exit()
                else:
                    if is_file:
                        stat = path.stat()
                        size = stat.st_size
                        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime('%Y-%m-%d-%H:%M:%S:%f')
                        ctime = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).strftime('%Y-%m-%d-%H:%M:%S:%f')

                        name = f'{name}\t{size}\t{mtime}\t{ctime}'

            yield prefix + pointer + name

            try:
                is_dir = path.is_dir()
            except OSError as e:
                if verbose:
                    print(f'skipping {str(path)}: {e}')
                continue
            except BaseException as e:
                print(f'file access error :: {str(path)}: {e}')
                input('press any key to exit')
                exit()
            else:
                if is_dir:  # extend the prefix and recurse:
                    if exclude_links and path.is_symlink():
                        continue
                    extension = branch if pointer == tee else space
                    # i.e. space because last, └── , above so no more |
                    yield from recursive_tree(path, prefix=prefix + extension)

    with open(dst, 'w', encoding="utf-8") as fid:
        # tree_str = ''
        for line in tqdm(recursive_tree(src), desc=dst):
            fid.write(line + '\n')
            # tree_str += line + '\n'
            # print(line)


def main():
    # params = Params()
    params = paramparse.process(Params)  # type: Params

    assert os.path.exists(params.src), "src does not exist: {}".format(params.src)

    f1 = params.dst
    cmp = params.dst + '.cmp'

    if os.path.exists(params.dst):
        f2 = params.dst + '.backup'
        shutil.move(f1, f2)
    else:
        f2 = None

    # tree_cmd = 'tree {} /a /f > {}'.format(params.src, params.dst)
    # os.system(tree_cmd)

    tree_to_file(params.src, params.dst, params.exclude_links, params.extra_info, params.verbose)

    if f2 is not None:

        file1 = open(f1, 'r',
                     encoding="utf-8"
                     ).readlines()
        file2 = open(f2, 'r',
                     encoding="utf-8"
                     ).readlines()

        diffs = difflib.unified_diff(file2, file1, n=0)

        # htmlDiffer = difflib.HtmlDiff()
        # htmldiffs = htmlDiffer.make_file(file1, file2)

        diffs = list(diffs)

        # vol_serial_number_lines = [k for k in diffs
        #                            if k.startswith('-Volume serial number') or
        #                            k.startswith('+Volume serial number')]
        # if vol_serial_number_lines:
        #     vol_serial_number_idx = max(diffs.index(k) for k in vol_serial_number_lines)
        #     if vol_serial_number_idx == len(diffs) - 1:
        #         diffs = []
        #     else:
        #         diffs = diffs[vol_serial_number_idx + 1:]

        if diffs:
            with open(cmp, 'w',
                      encoding="utf-8"
                      ) as outfile:
                # outfile.write('{}\t{}\n'.format(f2, f1))
                for diff in diffs:
                    outfile.write(diff)

            # subprocess.call("start " + cmp, shell=True)
            os.startfile(cmp)


if __name__ == '__main__':
    main()
