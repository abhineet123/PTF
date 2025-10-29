import numpy as np
import re


class IndexedStr(str):
    row_id = None
    col_id = None


def from_diagonals(diagonals, shape, dtype):
    """reconstruct array from list of diagonals
    adapted from: https://stackoverflow.com/a/78605058
    """
    arr = np.empty(shape, dtype)
    height, width = shape
    for h in range(height):
        arr[h, :] = [
            diagonal[h if d < width else h - (d - (width - 1))]
            for d, diagonal in enumerate(diagonals[h:h + width], start=h)
        ]
    return np.fliplr(arr)


def add_marker(arr_str, arr_idx, sep, marker, search_len):
    diags_idxs_all = [x for xs in [list(range(idx, idx + search_len)) for idx in arr_idx] for x in xs]
    diags_non_idxs = list(set(range(len(arr_str))) - set(diags_idxs_all))
    diags_arr = np.array(list(arr_str))
    diags_arr[diags_non_idxs] = marker

    sep_idxs = [m.start() for m in re.finditer(sep, arr_str)]
    diags_arr[sep_idxs] = sep
    diags_str_marked = ''.join(diags_arr)
    diags_marked = diags_str_marked.split(sep)
    return diags_marked


def find_all_occurrences(diags_str, search_str):
    """
    find all occurrences of a substring with overlap and reversal
    adapted from: https://stackoverflow.com/a/4664889
    """
    diags_idx = [m.start() for m in re.finditer(f'(?={search_str}|{search_str[::-1]})', diags_str)]
    return diags_idx


def find_linear_occurrences(arr, search_term, sep, marker):
    arr_1d = np.array([''.join(word) for word in arr])
    arr_str = sep.join(arr_1d)
    arr_idx = find_all_occurrences(arr_str, search_term)

    str_marked = add_marker(arr_str, arr_idx, sep, marker, len(search_term))
    arr_marked = np.array([list(word) for word in str_marked])
    return arr_idx, arr_marked


def find_diagonal_occurrences(arr, search_term, sep, marker):
    h, w = arr.shape
    diags = [''.join(arr.diagonal(offset=k)) for k in range(w - 1, -h, -1)]

    arr_rec = from_diagonals(diags, arr.shape, arr.dtype)
    assert np.array_equal(arr, arr_rec)

    diags_str = sep.join(diags)

    diags_idx = find_all_occurrences(diags_str, search_term)

    diags_marked = add_marker(diags_str, diags_idx, sep, marker, len(search_term))
    arr_marked = from_diagonals(diags_marked, arr.shape, arr.dtype)

    return diags_idx, arr_marked


def load_input_text(input_file):
    input_1d = np.loadtxt(input_file, dtype=str)
    input_2d = np.array([list(word) for word in input_1d])
    for row_id, row in enumerate(input_2d):
        for col_id, col in enumerate(row):
            col.row_id = row_id
            col.col_id = col_id

    return input_1d, input_2d


def solve_part_2():
    search_term = 'MAS'
    idx_str = np.dtype([('val', np.str_, 16), ('idx', np.int32, (2,))])
    input_1d, input_2d = load_input_text(input_file)



def solve_part_1(input_file):
    search_term = 'XMAS'

    sep = '-'
    marker = '.'

    input_1d, input_2d = load_input_text(input_file)

    input_t_2d = input_2d.transpose()

    """horizontal"""
    horz_idx, horz_marked = find_linear_occurrences(input_2d, search_term, sep, marker)

    """vertical"""
    vert_idx, vert_marked = find_linear_occurrences(input_t_2d, search_term, sep, marker)
    vert_marked = vert_marked.transpose()

    """left-to-right diagonals"""
    lr_idx, lr_marked = find_diagonal_occurrences(input_2d, search_term, sep, marker)

    """right-to-left diagonals"""
    input_flip_2d = np.fliplr(input_2d)
    rl_idx, rl_marked = find_diagonal_occurrences(input_flip_2d, search_term, sep, marker)
    rl_marked = np.fliplr(rl_marked)

    all_marked = np.full_like(input_2d, marker)
    for marked in [horz_marked, vert_marked, lr_marked, rl_marked]:
        mask = marked != marker
        all_marked[mask] = marked[mask]

    all_marked_str = '\n'.join([''.join(word) for word in all_marked])

    total_count = sum(len(k) for k in [horz_idx, vert_idx, lr_idx, rl_idx])
    print(f'total_count: {total_count}')


if __name__ == '__main__':
    input_file = "ceres_input.txt"
    # input_file = "ceres_input_small.txt"
    solve_part_1(input_file)
    # solve_part_2(input_file)
