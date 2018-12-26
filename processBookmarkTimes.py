import sys
import os
from datetime import datetime, timedelta
import pyperclip
from Tkinter import Tk

in_txt = Tk().clipboard_get()

lines = in_txt.split('\n')
data = [line.split(' ') for line in lines  if line.strip()]
out_lines = ['{} {}'.format(item[0], item[1].replace('-', ':')) for item in data]
out_txt = ''
for line in out_lines:
    out_txt = line if not out_txt else '{}\n{}'.format(out_txt, line)
print(out_txt)
try:
    pyperclip.copy(out_txt)
    spam = pyperclip.paste()
except pyperclip.PyperclipException as e:
    print('Copying to clipboard failed: {}'.format(e))


