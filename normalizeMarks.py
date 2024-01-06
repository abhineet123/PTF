import os
import sys
import pandas as pd
import numpy as np
from pprint import pformat
from openpyxl import load_workbook
from datetime import datetime

# marks_file_path = 'A2/A2_Marks_hager.xlsx'
# marks_file_path = 'A4/A4_Marks.xlsx'
# marks_file_path = 'A7/A7_Marks.xlsx'
# marks_file_path = 'A8/A8_Marks.xlsx'
marks_file_path = 'A9/A9_Marks_quiz.xlsx'
sheet_name = 'proc'
# sheet_name = 'raw'
out_sheet_name = ''
# out_sheet_name = 'scaled3'
# cols = ['Q1 (20)', 'Q2 (20)', 'Q3 (30)', 'Q4 (30)', 'Total']
max_vals = [25, ]
# max_vals = [20, 20, 30, 30, 100]
cols = ['Total']
n_decimals = 1

data = pd.read_excel(marks_file_path, sheet_name=sheet_name)
df = pd.DataFrame(data)

markers = df.Marker.unique()
markers = [k for k in markers if isinstance(k, str)]

# sorted_df = df.sort_values(by=['Marker'])
# print(sorted_df)
print(markers)

mean_dict = {}
std_dict = {}

updated_marks = []

out_dict = {}

out_txt = 'Marker'
for col in cols:
    out_txt = '{}\t{}\t{}'.format(out_txt, col + ' mean', col + ' std')

for marker in markers:
    out_txt = '{}\n{}'.format(out_txt, marker)

    marker_marks = df[df.Marker == marker]
    # print('\nmarker: {}'.format(marker))
    print(marker_marks)

    out_dict[marker] = {}

    for col in cols:
        col_val = marker_marks[col]
        mean = np.mean(col_val)
        std = np.std(col_val)

        # print('col: {}'.format(col))
        # print('mean: {}'.format(mean))
        # print('std: {}'.format(std))

        out_dict[marker][col] = {
            'mean': mean,
            'std': std,
        }
        if col not in mean_dict:
            mean_dict[col] = []
            std_dict[col] = []

        mean_dict[col].append(mean)
        std_dict[col].append(std)

        norm_col_val = (col_val - mean) / std

        marker_marks['norm {}'.format(col)] = norm_col_val

        out_dict[marker][col] = {
            'mean': mean,
            'std': std,
        }

        out_txt = '{}\t{:.3f}\t{:.3f}'.format(out_txt, mean, std)

    updated_marks.append(marker_marks)

updated_marks = pd.concat(updated_marks)

out_dict['average'] = {}
out_txt = '{}\n{}'.format(out_txt, 'average')

for col_id, col in enumerate(cols):
    avg_mean = np.mean(mean_dict[col])
    avg_std = np.mean(std_dict[col])

    out_txt = '{}\t{:.3f}\t{:.3f}'.format(out_txt, avg_mean, avg_std)

    max_val = max_vals[col_id]

    out_dict['average'][col] = {
        'mean': avg_mean,
        'std': avg_std,
    }
    # print('updated_marks:\n{}'.format(updated_marks))

    norm_col_val = updated_marks['norm {}'.format(col)]
    col_val = updated_marks[col]

    scaled_col_val = norm_col_val * avg_std + avg_mean
    scaled_col_val[col_val == 0] = 0
    scaled_col_val[col_val == max_val] = max_val

    scaled_col_val[scaled_col_val < 0] = 0
    scaled_col_val[scaled_col_val > max_val] = max_val

    scaled_col_val = np.round(scaled_col_val, decimals=n_decimals)

    updated_marks['scaled {}'.format(col)] = scaled_col_val

# print('avg_mean: {}'.format(avg_mean))
# print('avg_std: {}'.format(avg_std))

# print(pformat(out_dict))

out_txt = '{}\n{}'.format(out_txt, 'std')

for col_id, col in enumerate(cols):
    avg_mean = np.std(mean_dict[col])
    avg_std = np.std(std_dict[col])

    out_txt = '{}\t{:.3f}\t{:.3f}'.format(out_txt, avg_mean, avg_std)

print(out_txt)

# print('updated_marks:\n{}'.format(updated_marks))

if out_sheet_name:
    book = load_workbook(marks_file_path)
    writer = pd.ExcelWriter(marks_file_path, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    updated_marks.to_excel(writer, sheet_name=out_sheet_name)
    writer.save()
else:
    marks_file_dir = os.path.dirname(marks_file_path)
    marks_file_name = os.path.basename(marks_file_path)
    marks_file_name_no_ext, marks_file_ext = os.path.splitext(marks_file_name)
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    out_file_name = '{}_{}{}'.format(marks_file_name_no_ext, time_stamp, marks_file_ext)
    out_file_path = os.path.join(marks_file_dir, out_file_name)

    print('out_file_path: {}'.format(out_file_path))

    updated_marks.to_excel(out_file_path, sheet_name='scaled')
