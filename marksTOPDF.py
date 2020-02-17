import os
import sys
import math
import pandas as pd
import numpy as np
from pprint import pformat
from openpyxl import load_workbook
from datetime import datetime

import win32com.client

excel_app = win32com.client.Dispatch("Excel.Application")
excel_app.Visible = False

# sys.exit()

# root_dir = ''
root_dir = 'H:/UofA/206_W20'

marks_file_path = 'A1/A1_Marks.xlsx'
print_file_path = 'A1/A1_Marks_print.xlsx'
out_dir = 'A1/auto_comments'

'''
folder containing all extracted submissions
'''
# participant_ids = ''
submissions_dir = 'A1/submissions'

if root_dir:
    marks_file_path = os.path.join(root_dir, marks_file_path)
    print_file_path = os.path.join(root_dir, print_file_path)
    out_dir = os.path.join(root_dir, out_dir)

if submissions_dir:
    if root_dir:
        submissions_dir = os.path.join(root_dir, submissions_dir)

    participant_ids = [k for k in os.listdir(submissions_dir) if
                       os.path.isdir(os.path.join(submissions_dir, k))]
    participant_ids_list = [k.split('_', 1) for k in participant_ids]

    participant_ids_dict = {name: _id for name, _id in participant_ids_list}
else:
    participant_ids_dict = {}

# sheet_name = 'proc'
sheet_name = 'raw'
out_row_id = 4
id_col = 'Student ID'
_id = 0
# _id = 1594189
# marks_file_path = "H:/UofA/411 F18/A2/A2_Marks.xlsx"
# print_file_path = "H:/UofA/411 F18/A2/A2_Marks_print.xlsx"
# out_dir = 'H:/UofA/411 F18/A2/auto_comments'
# sheet_name = 'Sheet1'
# out_row_id = 5
# id_col = 'Name'


name_col = 'Name'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

print('Reading marks from {} and print template from {}'.format(marks_file_path, print_file_path))

data = pd.read_excel(marks_file_path, sheet_name=sheet_name)
df = pd.DataFrame(data)

ids = df[id_col].unique()

# print(df.columns)
# print(df[id_col])

ids = [k for k in ids if isinstance(k, str) or not math.isnan(k)]
# ids = [ids[0], ]

if _id:
    ids = [_id, ]

for _id in ids:
    _marks = df[df[id_col] == _id]

    name = _marks.iloc[0][name_col]

    book = load_workbook(print_file_path)
    sheet = book.active

    # out_row = []
    out_txt = ''

    for col_id, col in enumerate(_marks.columns):
        cell_val = _marks.iloc[0][col]
        # out_row.append(cell_val)

        cell_id = '{}{}'.format(chr(ord('A') + col_id), out_row_id)
        sheet[cell_id] = cell_val
        out_txt = '{}\t{}'.format(out_txt, cell_val)

    out_txt = '{}'.format(out_txt)

    # print(pformat(_marks))

    print(out_txt)

    if participant_ids_dict:
        participant_id = participant_ids_dict[name]
        _out_dir = os.path.join(out_dir, '{}_{}'.format(name, participant_id))
        if not os.path.isdir(_out_dir):
            os.makedirs(_out_dir)
    else:
        _out_dir = out_dir

    # sheet.append(out_row)
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_xls_path = os.path.join(_out_dir, '{}_{}.xlsx'.format(name, time_stamp))
    out_xls_path = os.path.abspath(out_xls_path)

    book.save(out_xls_path)
    # print('Done saving {}'.format(out_xls_path))

    wb = excel_app.Workbooks.Open(out_xls_path)

    ws_index_list = [1, ]  # say you want to print these sheets



    if id_col != name_col:
        out_pdf_name = '{}_{:d}.pdf'.format(name, int(_id))
    else:
        out_pdf_name = '{}.pdf'.format(name)

    out_pdf_path = os.path.join(_out_dir, out_pdf_name)
    out_pdf_path = os.path.abspath(out_pdf_path)

    wb.WorkSheets(ws_index_list).Select()
    ws = wb.ActiveSheet
    ws.PageSetup.FitToPagesWide = 1
    ws.ExportAsFixedFormat(0, out_pdf_path)

    print('Saving PDF to {}\n'.format(out_pdf_path))
    excel_app.Workbooks.Close()

    os.remove(out_xls_path)

excel_app.Quit()
