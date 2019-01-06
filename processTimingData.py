from datetime import datetime, timedelta
import pyperclip
from Tkinter import Tk

in_txt = Tk().clipboard_get()

lines = in_txt.split('\n')
lines = [line for line in lines if line.strip()]
start_t = None
curr_t = None
out_txt = ''
for line in lines:
    if line.startswith('Timing Data'):
        continue
    if line.startswith('Timing Description: Start Time: '):
        _line = line.replace('Timing Description: Start Time: ', '').strip()
        start_t = datetime.strptime(_line, '%b %d, %Y %I:%M:%S %p')
        out_txt += start_t.strftime('%d/%m/%Y') + '\n'
        # print('start_t: {}'.format(start_t))
    if line.startswith('Lap Description: '):
        _line = line.replace('Lap Description: ', '').strip()
        if ' <Stopped>' in _line:
            _line = _line.replace(' <Stopped>', '')
        out_txt += _line.replace('...', '\t')
    elif line.startswith('Lap Time: '):
        _line = line.replace('Lap Time: ', '').strip()
        _line_data = _line.split('.')
        # print('_line_data: {}'.format(_line_data))

        _line = '{}:{}'.format(_line_data[0], int(_line_data[1]) * 10000)
        # print('_line: {}'.format(_line))

        lap_t = datetime.strptime(_line, '%H:%M:%S:%f')
        # print('curr_t: {}'.format(lap_t))
        out_txt += lap_t.strftime('%H:%M:%S') + '.{}\t'.format(_line_data[1])
    elif line.startswith('Lap Total Time: '):
        _line = line.replace('Lap Total Time: ', '').strip()
        _line_data = _line.split('.')
        # print('_line_data: {}'.format(_line_data))

        _line = '{}:{}'.format(_line_data[0], int(_line_data[1]) * 10000)
        # print('_line: {}'.format(_line))

        lap_total_t = datetime.strptime(_line, '%H:%M:%S:%f')
        # print('curr_t: {}'.format(lap_t))
        out_txt += lap_total_t.strftime('%H:%M:%S') + '.{}\t'.format(_line_data[1])
        curr_t = start_t + timedelta(hours=lap_total_t.hour, minutes=lap_total_t.minute,
                                     seconds=lap_total_t.second, microseconds=lap_total_t.microsecond)
        out_txt += curr_t.strftime('%H:%M:%S') + '.{}\n'.format(int(curr_t.microsecond / 10000))

print(out_txt)
# with open(out_fname, 'w') as out_fid:
#     out_fid.write(out_txt)
try:
    pyperclip.copy(out_txt)
    spam = pyperclip.paste()
except pyperclip.PyperclipException as e:
    print('Copying to clipboard failed: {}'.format(e))
