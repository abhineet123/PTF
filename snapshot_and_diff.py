# import admin

# if not admin.isUserAdmin():
#     admin.runAsAdmin()

import os
import difflib
import shutil
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone
# import gzip
import subprocess

import paramparse

from Misc import linux_path


class Params:
    def __init__(self):
        self.src_label = ''
        self.src = ''
        self.dst = ''
        self.excluded_names = ''
        self.exclude_links = 1
        self.extra_info = 1
        self.verbose = 1


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

def recursive_listdir(src, verbose, err_file, exclude_links, excluded_names):
    try:
        contents = list(Path(src).iterdir())
    except PermissionError as e:
        print(f'\n{e}\n')
        # input('press any key to continue')
        return

    for path in contents:
        is_valid = False

        try:
            is_dir = path.is_dir()
        except OSError as e:
            if verbose:
                print(f'\nskipping {str(path)}: {e}\n')

                # input('press any key to continue')
        except BaseException as e:
            with open(err_file, 'a') as err_fid:
                err_fid.write(f'{e}')
            input('press any key to continue')
            exit()
        else:
            is_valid = True

        yield str(path)

        if is_valid and is_dir:  # extend the prefix and recurse:
            if excluded_names and any(k in str(path) for k in excluded_names):
                print(f'\nexcluding: {str(path)}\n')
                continue

            if exclude_links and is_link(str(path)):
                print(f'\nexcluding symlink: {str(path)}\n')
                continue

            yield from recursive_listdir(path, verbose, err_file, exclude_links, excluded_names)


def tree_to_file(src, dst, exclude_links, extra_info, err_file, verbose):
    src = Path(src)
    # prefix components:
    space = '    '
    branch = '│   '
    # pointers:
    tee = '├── '
    last = '└── '

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    def recursive_tree(dir_path: Path,
                       prefix: str = '',
                       ):
        """A recursive generator, given a directory Path object
        will yield a visual tree structure line by line
        with each line prefixed by the same characters
        """
        try:
            contents = list(dir_path.iterdir())
        except PermissionError as e:
            print(f'\n{e}\n')
            # input('press any key to continue')
            return

        # contents each get pointers that are ├── with a final └── :
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            name = path.name

            is_valid = False

            try:
                is_dir = path.is_dir()
            except OSError as e:
                if verbose:
                    print(f'\nskipping {str(path)}: {e}\n')

                    # input('press any key to continue')
            except BaseException as e:
                err_txt = f'{time_stamp}\n{e}\n'
                print(err_txt)
                with open(err_file, 'a') as err_fid:
                    fid.write(err_txt)
                input('press any key to continue')
                exit()
            else:
                is_valid = True

            if is_valid and extra_info and not is_dir:
                try:
                    stat = path.stat()
                except FileNotFoundError as e:
                    err_txt = f'{time_stamp}\n{e}\n'
                    print(err_txt)
                    with open(err_file, 'a') as err_fids:
                        fid.write(err_txt)
                    # input('press any key to continue')
                    continue

                size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime('%Y-%m-%d-%H:%M:%S:%f')
                ctime = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).strftime('%Y-%m-%d-%H:%M:%S:%f')

                name = f'{name}\t{size}\t{mtime}\t{ctime}'

            yield prefix + pointer + name

            if is_valid and is_dir:  # extend the prefix and recurse:
                if exclude_links and is_link(str(path)):
                    print(f'\nexcluding symlink: {str(path)}\n')
                    continue

                extension = branch if pointer == tee else space
                # i.e. space because last, └── , above so no more |
                yield from recursive_tree(path, prefix=prefix + extension)

    # with open(dst, 'w', encoding="utf-8") as fid:

    if dst.endswith('.html'):
        title = os.path.splitext(os.path.basename(dst))
        cmd = f'Snap2HTMl -path:"{src}" -outfile:"{dst}" -title:"{title}" -hidden -system'
        print(f'running: {cmd}')
        os.system(cmd)
    else:
        with open(dst, 'w', encoding="utf-8") as fid:
            # tree_str = ''
            for line in tqdm(recursive_tree(src), desc=dst):
                out_line = line + '\n'
                fid.write(out_line)
                # fid.write(out_line.encode('utf-8'))
                # tree_str += line + '\n'
                # print(line)


def is_link(src):
    child = subprocess.Popen(
        "fsutil reparsepoint query \"{}\"".format(src),
        stdout=subprocess.PIPE
    )
    streamdata = child.communicate()[0]
    rc = child.returncode

    if rc == 0:
        return True
    return False


def main():
    # params = Params()
    params = paramparse.process(Params)  # type: Params

    f1 = params.dst
    # cmp = params.dst + '.cmp'
    title, ext = os.path.splitext(os.path.basename(params.dst))
    dst_dir = os.path.dirname(params.dst)
    dst_parent_dir = os.path.dirname(dst_dir)

    err_file = params.dst.replace(ext,  '.errors')

    backup_dst_dir = linux_path(dst_parent_dir, 'backup')
    cmp_dst_dir = linux_path(dst_parent_dir, 'cmp')

    os.makedirs(backup_dst_dir, exist_ok=1)
    os.makedirs(cmp_dst_dir, exist_ok=1)

    cmp_fname = title + '.cmp'
    cmp = linux_path(cmp_dst_dir, cmp_fname)

    if params.src_label:
        import psutil
        import win32api

        partitions = psutil.disk_partitions()
        for partition in partitions:
            partition_info = win32api.GetVolumeInformation(partition.device)

            partition_label = partition_info[0]

            if partition_label == params.src_label:
                params.src = partition.device
                break
        else:
            raise AssertionError(f'src_label: {params.src_label} not found')

    assert os.path.exists(params.src), "src does not exist: {}".format(params.src)

    if params.dst.endswith('.html'):
        if os.path.exists(f1):
            backup_fname = title + '_backup' + ext
            f2 = linux_path(backup_dst_dir, backup_fname)
            shutil.move(f1, f2)
        else:
            f2 = None

        cmd = f'Snap2HTMl -path:"{params.src}" -outfile:"{params.dst}" -title:"{title}" -hidden -system'

        print(f'running: {cmd}')
        os.system(cmd)

        if f2 is not None:
            file1 = open(f1, 'r', encoding="utf-8").readlines()
            file2 = open(f2, 'r', encoding="utf-8").readlines()

        # excluded = ['This file was generated by Snap2HTML',
        #             '<div class="app_header_stats">']
    else:
        # f1 += '.gz'

        cmp = params.dst + '.cmp'
        backup_fname = title + '_backup' + ext
        f2 = linux_path(backup_dst_dir, backup_fname)

        if os.path.exists(f1):
            backup_fname = title + '_backup' + ext
            f2 = linux_path(backup_dst_dir, backup_fname)
            shutil.move(f1, f2)
            # f2 += '.gz'
        elif not os.path.exists(f2):
            f2 = None

        if params.dst.endswith('.bin'):

            # tree_cmd = 'tree {} /a /f > {}'.format(params.src, params.dst)
            # os.system(tree_cmd)

            exclude_links = params.exclude_links
            excluded_names = params.excluded_names
            verbose = params.verbose
            files_list = []

            if excluded_names:
                assert os.path.isfile(excluded_names), f"invalid excluded_names file: {excluded_names}"
                excluded_names = open(excluded_names, 'r').read().splitlines()

            pbar = tqdm(recursive_listdir(params.src, verbose, err_file, exclude_links, excluded_names), position=0, leave=True)

            for path in tqdm(pbar):
                pbar.set_description(f"{path[:40]}...")
                files_list.append(path)

            # files_gen = [[os.path.join(dirpath, f) for f in filenames ]
            #                   for (dirpath, dirnames, filenames) in
            #              tqdm(os.walk(params.src, followlinks=True), desc='gen')]
            # files_list = [item for sublist in files_gen for item in tqdm(sublist, desc='list')]

            print(f'{title} :: {len(files_list)} files')

            import pickle
            with open(params.dst, 'wb') as f:
                pickle.dump(files_list, f, pickle.HIGHEST_PROTOCOL)

            if f2 is not None:
                with open(f2, 'rb') as f:
                    prev_files_list = pickle.load(f)

                print(f'{title} :: {len(prev_files_list)} prev_files')

                new_files = list(set(files_list) - set(prev_files_list))
                deleted_files = list(set(prev_files_list) - set(files_list))

                if new_files or deleted_files:

                    with open(cmp, 'w',
                              encoding="utf-8"
                              ) as outfile:
                        outfile.write(f'{title} :: {len(new_files)} new_files\n')
                        outfile.write(f'{title} :: {len(deleted_files)} deleted_files\n')

                        if new_files:
                            outfile.write('new_files:\n\n')
                            outfile.write('\n'.join(new_files) + '\n')

                        if deleted_files:
                            outfile.write('\n\ndeleted_files:\n\n')
                            outfile.write('\n'.join(deleted_files) + '\n')

                    os.startfile(cmp)
            return

        tree_to_file(params.src, f1, params.exclude_links, params.extra_info,
                     err_file, params.verbose)

        if f2 is not None:
            file1 = open(f1, 'r',
                         encoding="utf-8"
                         ).readlines()
            file2 = open(f2, 'r',
                         encoding="utf-8"
                         ).readlines()

            # file1 = gzip.open(f1, 'rt',
            #                   encoding="utf-8"
            #                   ).readlines()
            # file2 = gzip.open(f2, 'rt',
            #                   encoding="utf-8"
            #                   ).readlines()
        # excluded = []

    if f2 is None:
        return

    diffs = difflib.unified_diff(file2, file1, n=0)

    # diffs = [diff for diff in diffs if diff[1:].startswith('D.p')]

    # htmlDiffer = difflib.HtmlDiff()
    # htmldiffs = htmlDiffer.make_file(file1, file2)

    diffs = list(diffs)
    n_diffs = len(diffs)

    if params.dst.endswith('.html'):

        block_start_ids = [i for i, diff in enumerate(diffs) if diff.startswith('@@ ')]
        block_start_ids.insert(0, 0)

        block_end_ids = block_start_ids[1:]
        block_end_ids.append(n_diffs)

        blocks = [list(zip(range(s, e), diffs[s:e])) for s, e in zip(block_start_ids, block_end_ids)]

        # last_block = blocks[-1]
        # print()

        all_lines_to_exclude = []

        # block_diffs = []

        for block in blocks:
            lines_old = [k for k in block if k[1].startswith('-')]
            lines_new = [k for k in block if k[1].startswith('+')]

            n_lines_old = len(lines_old)
            n_lines_new = len(lines_new)

            if n_lines_old != n_lines_new:
                continue

            # line_diffs = []

            lines_to_exclude = []
            for data_old, data_new in zip(lines_old, lines_new):

                line_id_old, line_old = data_old
                line_id_new, line_new = data_new

                # if any(k in line_old for k in excluded):
                #     lines_to_exclude.append(line_id_old)

                # if any(k in line_new for k in excluded):
                #     lines_to_exclude.append(line_id_new)

                is_excluded = False

                if not line_old[1:].startswith('D.p'):
                    lines_to_exclude.append(line_id_old)
                    is_excluded = True

                if not line_new[1:].startswith('D.p'):
                    lines_to_exclude.append(line_id_new)
                    is_excluded = True

                if is_excluded:
                    continue

                list_old = line_old[1:].split(',')
                list_new = line_new[1:].split(',')

                n_sec = len(list_old)

                if len(list_new) != n_sec:
                    continue

                # list_diff = [
                #     # [li for li in difflib.ndiff(k1, k2) if li[0] != ' ']
                #     [k1, k2]
                #     if k1 != k2 else []
                #     for k1, k2 in zip(list_old, list_new)]

                diff_sec_ids = [i for i, k in enumerate(zip(list_old, list_new)) if k[0] != k[1]]

                if len(diff_sec_ids) == 1 and diff_sec_ids[0] == len(list_old) - 1:
                    lines_to_exclude.append(line_id_old)
                    lines_to_exclude.append(line_id_new)

                # line_diffs.append([line_old, line_new, list_diff])
                # print()

            n_lines_to_exclude = len(lines_to_exclude)
            if n_lines_to_exclude == 2 * n_lines_old:
                """all lines in block excluded so exclude its header too"""
                lines_to_exclude.insert(0, block[0][0])

            all_lines_to_exclude += lines_to_exclude

        diffs = [diff for i, diff in enumerate(diffs) if i not in all_lines_to_exclude]

        # proc_diffs = []
        # proc_diffs_no_prefix = []

        pos_entries = {}
        neg_entries = {}
        all_entries = {}
        entries_to_id = {}

        diff_id = 0

        for diff in diffs:
            if diff.startswith('+D.p'):
                is_pos = True
            elif diff.startswith('-D.p'):
                is_pos = False
            else:
                # proc_diffs.append('')
                continue

            diff = diff.replace('D.p', '').replace('"', '').replace('([', '').replace('])', '')

            diff_items = diff.split(',')

            assert len(diff_items) >= 3, f"invalid diff_items: {diff_items}"
            del diff_items[-1]
            del diff_items[-1]

            proc_diff_items = []
            for diff_item in diff_items:
                diff_item_list = diff_item.split('*')
                proc_diff_items.append(diff_item_list[0])

            root_item = proc_diff_items[0][1:]
            root_path = Path(root_item)

            if params.exclude_links and (
                    is_link(str(root_path)) or
                    any(is_link(str(k)) for k in root_path.parents)):
                continue

            if is_pos:
                curr_pos_entries = proc_diff_items[1:]
                try:
                    curr_neg_entries = neg_entries[root_item]
                except KeyError:
                    pass
                else:
                    unique_pos_entries = list(set(curr_pos_entries) - set(curr_neg_entries))
                    unique_neg_entries = list(set(curr_neg_entries) - set(curr_pos_entries))

                    curr_pos_entries = unique_pos_entries
                    neg_entries[root_item] = unique_neg_entries

                    all_entries['-' + root_item] = unique_neg_entries

                pos_entries[root_item] = curr_pos_entries

                all_entries['+' + root_item] = curr_pos_entries
                entries_to_id['+' + root_item] = diff_id

            else:
                neg_entries[root_item] = proc_diff_items[1:]
                entries_to_id['-' + root_item] = diff_id
                all_entries['-' + root_item] = proc_diff_items[1:]

            diff_id += 1

            # try:
            #     idx = proc_diffs_no_prefix.index(proc_diff[1:])
            # except ValueError:
            #     proc_diffs_no_prefix.append(proc_diff[1:])
            #     proc_diff = proc_diff.replace('\t', '\n\t')
            #     proc_diffs.append(proc_diff)
            # else:
            #     del proc_diffs_no_prefix[idx]
            #     del proc_diffs[idx]
        #

        proc_diffs = [None, ] * diff_id

        for root_item in all_entries:
            diff_id = entries_to_id[root_item]

            assert proc_diffs[diff_id] is None, f"duplicate diff_id found: {diff_id}"

            proc_diff_items = all_entries[root_item]

            if not proc_diff_items:
                continue

            proc_diff_items.insert(0, root_item)
            proc_diff = '\t'.join(proc_diff_items)
            proc_diff = proc_diff.replace('\t', '\n\t')

            proc_diffs[diff_id] = proc_diff

        diffs = proc_diffs

        # print()

        # vol_serial_number_lines = [k for k in diffs
        #                            if k.startswith('-Volume serial number') or
        #                            k.startswith('+Volume serial number')]
        # if vol_serial_number_lines:
        #     vol_serial_number_idx = max(diffs.index(k) for k in vol_serial_number_lines)
        #     if vol_serial_number_idx == len(diffs) - 1:
        #         diffs = []
        #     else:
        #         diffs = diffs[vol_serial_number_idx + 1:]

        # diffs = [diff for diff in diffs if all(k not in diff for k in excluded)]

        # dummy_starts = [
        #     '---',
        #     '+++',
        #     '@@',
        # ]
    """only diff lines corresponding to stats info remain"""
    # if all(any(diff.strip().startswith(k) for k in dummy_starts) for diff in diffs):
    #     diffs = []

    if diffs:
        with open(cmp, 'w',
                  encoding="utf-8"
                  ) as outfile:
            # outfile.write('{}\t{}\n'.format(f2, f1))
            for diff in diffs:
                # if any(k in diff for k in excluded):
                #     continue

                if diff is None:
                    continue

                outfile.write(diff + '\n')

        # subprocess.call("start " + cmp, shell=True)
        # input('press any key to show diffs')
        os.startfile(cmp)


if __name__ == '__main__':
    main()
    # try:
    #     main()
    # except BaseException as e:
    #     print(f'\n{e}\n')
    #     input('press any key to exit')
