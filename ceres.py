import numpy as np
import re


def from_diagonals(diagonals, shape, dtype):
    """reconstruct array from diagonals
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


def add_marker(diags_str, diags_idx, sep, marker, search_len):
    sep_idxs = [m.start() for m in re.finditer(sep, diags_str)]
    diags_idxs_all = [x for xs in [list(range(idx, idx + search_len)) for idx in diags_idx] for x in xs]
    diags_non_idxs = list(set(range(len(diags_str))) - set(diags_idxs_all))
    diags_arr = np.array(list(diags_str))
    diags_arr[diags_non_idxs] = marker
    diags_arr[sep_idxs] = sep
    diags_str_marked = ''.join(diags_arr)
    diags_marked = diags_str_marked.split(sep)
    return diags_marked


def find_all_occurrences_with_overlap(diags_str, search_str):
    """
    adapted from:
    https://stackoverflow.com/a/4664889
    """
    diags_idx = [m.start() for m in re.finditer(f'(?={search_str}|{search_str[::-1]})', diags_str)]
    return diags_idx


def find_linear_occurrences(arr, sep, marker, search_term):
    arr_1d = np.array([''.join(word) for word in arr])
    arr_str = sep.join(arr_1d)
    arr_idx = find_all_occurrences_with_overlap(arr_str, search_term)

    str_marked = add_marker(arr_str, arr_idx, sep, marker, len(search_term))
    arr_marked = np.array([list(word) for word in str_marked])
    return arr_idx, arr_marked


def find_diagonal_occurences(arr, sep, marker, search_term):
    h, w = arr.shape
    diags = [''.join(arr.diagonal(offset=k)) for k in range(w - 1, -h, -1)]

    arr_rec = from_diagonals(diags, arr.shape, arr.dtype)
    assert np.array_equal(arr, arr_rec)

    diags_str = sep.join(diags)

    diags_idx = find_all_occurrences_with_overlap(diags_str, search_term)

    diags_marked = add_marker(diags_str, diags_idx, sep, marker, len(search_term))
    arr_marked = from_diagonals(diags_marked, arr.shape, arr.dtype)

    return diags_idx, arr_marked


def main():
    search_term = 'XMAS'
    input_file = "ceres_input.txt"
    # input_file = "ceres_input_small.txt"

    sep = '-'
    marker = '.'

    input_1d = np.loadtxt(input_file, dtype=str)
    input_2d = np.array([list(word) for word in input_1d])
    input_t_2d = input_2d.transpose()

    """horizontal"""
    horz_idx, horz_marked = find_linear_occurrences(input_2d, sep, marker, search_term)

    """vertical"""
    vert_idx, vert_marked = find_linear_occurrences(input_t_2d, sep, marker, search_term)
    vert_marked = vert_marked.transpose()

    """left-to-right diagonals"""
    lr_idx, lr_marked = find_diagonal_occurences(input_2d, sep, marker, search_term)

    """right-to-left diagonals"""
    input_flip_2d = np.fliplr(input_2d)
    rl_idx, rl_marked = find_diagonal_occurences(input_flip_2d, sep, marker, search_term)
    rl_marked = np.fliplr(rl_marked)

    all_marked = np.full_like(input_2d, '.')
    all_marked[horz_marked != marker] = horz_marked[horz_marked != marker]
    all_marked[vert_marked != marker] = vert_marked[vert_marked != marker]
    all_marked[lr_marked != marker] = lr_marked[lr_marked != marker]
    all_marked[rl_marked != marker] = rl_marked[rl_marked != marker]

    all_marked_str = '\n'.join([''.join(word) for word in all_marked])

    total_count = sum(len(k) for k in [horz_idx, vert_idx, lr_idx, rl_idx])
    print(f'total_count: {total_count}')


if __name__ == '__main__':
    main()
