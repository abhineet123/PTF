import sys
import os
from Misc import processArguments

params = {
    'filter_strings': ['magnet'],
    'url_fname': 'TED_urls.txt',
    # 'in_fname': 'TED_done.txt',
    'in_fname': 'TED_not_done_unique.txt',
    # 'in_fname': 'TED_not_done.txt',
    'out_fname': 'filtered.txt',
    # 0: only at start
    # 1: anywhere
    'mode': 1,
    'filter_type': 0,
    'retain_filtered': 1,
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)

filter_strings = params['filter_strings']
filter_type = params['filter_type']
in_fname = params['in_fname']
out_fname = params['out_fname']
url_fname = params['url_fname']
retain_filtered = params['retain_filtered']
mode = params['mode']

if mode == 0:
    not_done_lines = open(in_fname, 'r').readlines()
    not_done_lines = [k.strip() for k in not_done_lines]

    part_lines = [k for k in not_done_lines if k.endswith('.part')]
    mp4_lines = [k for k in not_done_lines if k.endswith('.mp4')]

    part_lines_no_ext = ['.'.join(k.split('.')[:-2]) for k in part_lines]
    mp4_lines_no_ext = ['.'.join(k.split('.')[:-1]) for k in mp4_lines]

    assert len(part_lines_no_ext) + len(mp4_lines_no_ext) == len(not_done_lines), "Mismatch in split"

    part_lines_no_ext_unique = list(set(part_lines_no_ext))
    mp4_lines_no_ext_unique = list(set(mp4_lines_no_ext))

    out_file = in_fname + '.out'
    with open(out_file, 'w') as out_fid:
        out_fid.write('\n'.join(part_lines_no_ext_unique + mp4_lines_no_ext_unique))
elif mode == 1:

    not_done_lines = open(in_fname, 'r').readlines()
    not_done_lines = [k.strip() for k in not_done_lines]
    not_done_lines = [k.split(',')[0] for k in not_done_lines]


    url_lines = open(url_fname, 'r', encoding="utf8").readlines()
    url_lines_proc = [k.strip().split('\t') for k in url_lines if k.strip()]
    urls, labels = map(list, zip(*url_lines_proc))

    labels_proc = [k.replace(':', '').replace('?', '').replace('"', ' ').replace('|', ' ').replace('/', ' ').strip()
                   for k in labels]
    # labels_proc_cp1252 = [str(k.encode(encoding='cp1252')) for k in labels]

    # out_file = url_fname + '.labels_proc'
    # with open(out_file, 'w') as out_fid:
    #     out_fid.write('\n'.join(labels_proc_cp1252))

    indices = [labels_proc.index(k) for k in not_done_lines]

    # invalid_lines = [not_done_lines[i] for i, idx in enumerate(indices) if idx == -1]

    out_file = url_fname + '.urls_labels_not_done'
    urls_labels_not_done = ['\t'.join((urls[i], labels[i])) for i in indices]
    with open(out_file, 'w') as out_fid:
        out_fid.write('\n'.join(urls_labels_not_done))





