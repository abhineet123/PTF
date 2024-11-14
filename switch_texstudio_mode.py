import os
import paramparse

class Params:
    ini_path = ""
    dark_x11_style = 'Adwaita Dark (txs)'
    light_x11_style = 'Adwaita (txs)'
    dark_gui_style = '2'
    light_gui_style = '1'
    dark_invert_status = 'true'
    light_invert_status = 'false'

def main():
    params = Params()
    paramparse.process(params)

    ini_path = params.ini_path

    assert ini_path, "ini_path must be provided"

    # import fileinput
    # for line in fileinput.input("test.txt", inplace=True):

    out_lines = []
    ini_lines = open(ini_path, 'r').readlines()
    is_dark_x11 = is_dark_gui = is_dark_invert = None
    for line in ini_lines:
        line = line.strip()
        if line.startswith('X11\Style='):
            x11_style = line.split('=')[1]
            if x11_style == params.dark_x11_style:
                is_dark_x11 = True
                line = line.replace(params.dark_x11_style, params.light_x11_style)
            elif x11_style == params.light_x11_style:
                is_dark_x11 = False
                line = line.replace(params.light_x11_style, params.dark_x11_style)
            else:
                raise AssertionError(f'invalid X11 style: {x11_style}')

            assert is_dark_gui is None or is_dark_x11 == is_dark_gui, "x11 and gui mismatch"
            assert is_dark_invert is None or is_dark_x11 == is_dark_invert, "x11 and invert mismatch"

        elif line.startswith('GUI\Style='):
            gui_style = line.split('=')[1]
            if gui_style == params.dark_gui_style:
                is_dark_gui = True
                line = line.replace(params.dark_gui_style, params.light_gui_style)
            elif gui_style == params.light_gui_style:
                is_dark_gui = False
                line = line.replace(params.light_gui_style, params.dark_gui_style)
            else:
                raise AssertionError(f'invalid gui style: {gui_style}')
            assert is_dark_x11 is None or is_dark_gui == is_dark_x11, "gui and x11 mismatch"
            assert is_dark_invert is None or is_dark_gui == is_dark_invert, "gui and invert mismatch"

        elif line.startswith('Preview\Invert%20Colors'):
            invert_status = line.split('=')[1]
            if invert_status == params.dark_invert_status:
                is_dark_invert = True
                line = line.replace(params.dark_invert_status, params.light_invert_status)
            elif invert_status == params.light_invert_status:
                is_dark_invert = False
                line = line.replace(params.light_invert_status, params.dark_invert_status)
            else:
                raise AssertionError(f'invalid invert_status: {invert_status}')
            assert is_dark_gui is None or is_dark_invert == is_dark_gui, "invert and gui mismatch"
            assert is_dark_x11 is None or is_dark_invert == is_dark_x11, "invert and x11 mismatch"

        out_lines.append(line)

    out_lines_str = '\n'.join(out_lines)
    open(ini_path, 'w').write(out_lines_str)

if __name__ == '__main__':
    main()
