import sys
import os
from pprint import pformat


def roman_to_int(s):
    """
    :type s: str
    :rtype: int
    """
    roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000, 'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90,
             'CD': 400, 'CM': 900}
    i = 0
    num = 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i + 2] in roman:
            num += roman[s[i:i + 2]]
            i += 2
        else:
            # print(i)
            num += roman[s[i]]
            i += 1
    return num


def main():
    file_path = 'Something Fresh - P. G. Wodehouse.txt'

    file_path = os.path.abspath(file_path)

    out_dir = os.path.dirname(file_path)
    out_fname = os.path.basename(file_path)
    out_fname_noext, out_fname_ext = os.path.splitext(out_fname)
    # chap_sep = ''
    chapter_sep = 'CHAPTER'
    convert_to_int = 1

    # fichier = open(file_path, "rb")
    # contentOfFile = fichier.read()
    # contentOfFile = contentOfFile.decode("utf-8")
    # asciidata = contentOfFile.encode("ascii", "ignore")
    #
    # fichier.close()
    #
    # out_file_path = file_path + '.ascii'
    #
    # fichierTemp = open(out_file_path, "w")
    # fichierTemp.write(asciidata)
    # fichierTemp.close()

    rep_path = 'C:/Users/Tommy/Documents/Backup/epub_txt_symbols_to_fix.txt'
    reps_lines = open(rep_path, "rb").read().decode("utf-8").split('\n')
    reps = [k.strip().split(' ') for k in reps_lines if k.strip()]

    print('reps_lines:\n{}'.format(reps_lines))
    print('reps : {}'.format(reps))

    non_ascii = [k[0] for k in reps]
    ascii = [k[1] for k in reps]
    ascii_ord = {k: ord(k) if len(k) == 1 else 0 for k in ascii}
    non_ascii_ord = {k: ord(k) if len(k) == 1 else 0 for k in non_ascii}

    print('non_ascii:\n{}'.format(pformat(non_ascii)))
    print('ascii:\n{}'.format(pformat(ascii)))
    print('non_ascii_ord:\n{}'.format(pformat(non_ascii_ord)))
    print('ascii_ord:\n{}'.format(pformat(ascii_ord)))

    in_data = open(file_path, "rb").read().decode("utf-8")
    in_lines = in_data.split('\n')

    out_lines = ''

    n_replaced = 0
    chap_out_file_path = ''
    chapter_id = ''
    for i, _line in enumerate(in_lines):
        out_line = ''
        for _char in _line:

            if ord(_char) >= 128:
                # try:
                #     _char.decode('ascii')
                # except UnicodeDecodeError:
                #     _char_utf8 = _char.decode('utf-8')
                try:
                    _idx = non_ascii.index(_char)
                except ValueError:
                    print('Non ASCII {} not found in replacement list in line {}:\n{}'.format(_char, i, _line))
                    sys.exit()
                else:
                    out_char = ascii[_idx]
                    # print('Replacing Non ASCII {} with {}'.format(_char, out_char))
                    out_char.replace('__sp__', ' ')
                    out_char.replace('__n__', '')
                    n_replaced += 1
            else:
                out_char = _char

            if any(ord(k) >= 128 for k in out_char):
                print('Non ASCII replacement for {} found: {} in line {}\n{}'.format(_char, out_char, i, _line))
                sys.exit()

            out_line += out_char

        if chapter_sep and out_line.startswith(chapter_sep):

            if out_lines and chap_out_file_path:
                print('Saving chapter {} to {}'.format(chapter_id, chap_out_file_path))
                with open(chap_out_file_path, 'w') as fid:
                    fid.write(out_lines)
                out_lines = ''

            _, chapter_id_str = out_line.strip().split(' ')
            if convert_to_int:
                chapter_id = str(roman_to_int(chapter_id_str))
                out_line = out_line.replace(chapter_id_str, chapter_id)
            else:
                chapter_id = chapter_id_str

            chap_out_file_path = os.path.join(out_dir, '{} chapter {}{}'.format(out_fname_noext, chapter_id,
                                                                                out_fname_ext))

        out_lines += out_line

    if out_lines and chap_out_file_path:
        print('Saving chapter {} to {}'.format(chapter_id, chap_out_file_path))
        with open(chap_out_file_path, 'w') as fid:
            fid.write(out_lines)
    else:
        out_file_path = file_path + '.ascii'
        with open(out_file_path, "w") as fid:
            fid.write(out_lines)

    print('Replaced {} characters'.format(n_replaced))


if __name__ == '__main__':
    main()
