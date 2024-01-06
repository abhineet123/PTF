import sys
import os
from datetime import datetime, timedelta
import pyperclip
from Tkinter import Tk

in_txt = Tk().clipboard_get()

lines = in_txt.split('\n')
data = [os.path.splitext(line)[0].split(' ') for line in lines if line.strip()]
out_lines = ['{} {}'.format(item[0], item[1].replace('-', ':')) if len(item) == 2 else '{}'.format(item[0]) for item in
             data]
out_txt = ''
for line in out_lines:
    out_txt = line if not out_txt else '{}\n{}'.format(out_txt, line)
print(out_txt)
try:
    pyperclip.copy(out_txt)
    spam = pyperclip.paste()
except pyperclip.PyperclipException as e:
    print('Copying to clipboard failed: {}'.format(e))
