import sys
from pprint import pformat

filePath = 'The Stuff of Thought_ Language - Steven Pinker.txt'

# fichier = open(filePath, "rb")
# contentOfFile = fichier.read()
# contentOfFile = contentOfFile.decode("utf-8")
# asciidata = contentOfFile.encode("ascii", "ignore")
#
# fichier.close()
#
# out_filePath = filePath + '.ascii'
#
# fichierTemp = open(out_filePath, "w")
# fichierTemp.write(asciidata)
# fichierTemp.close()

rep_path = 'epub_txt_symbols_to_fix.txt'
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

in_data = open(filePath, "rb").read().decode("utf-8")
in_lines = in_data.split('\n')

out_lines = ''

n_replaced = 0

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

    out_lines += out_line

out_filePath = filePath + '.ascii'
fichierTemp = open(out_filePath, "w")
fichierTemp.write(out_lines)
fichierTemp.close()

print('Replaced {} characters'.format(n_replaced))
