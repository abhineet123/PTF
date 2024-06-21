from datetime import datetime, timedelta
import paramparse

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk


def is_date(line):
    date_obj = None
    try:
        date_obj = datetime.strptime(line, '%Y-%m-%d')
    except ValueError:
        pass

    return date_obj


def is_time(line):
    time_found = 0
    time_obj = None
    try:
        time_obj = datetime.strptime(line, '%I:%M:%S %p')
    except ValueError:
        try:
            time_obj = datetime.strptime(line, '%I:%M %p')
        except ValueError:
            try:
                time_obj = datetime.strptime(line, '%H:%M:%S')
            except ValueError:
                pass
            # else:
        # else:
        #     temp2 = line.split(' ')
        #     _time, _pm = temp2
        #     line = '{}:00 {}'.format(_time, _pm)
        #     time_found = 1
    # else:
    #     time_found = 1

    if time_obj is not None:
        time_obj = time_obj.time()
        line = time_obj.strftime('%I:%M:%S %p')
        time_found = 1

    return line, time_found, time_obj


class Params:

    def __init__(self):
        self.add_comment = 1
        self.add_date = 1
        self.add_diff = 1
        self.categories = 1
        self.categories_out = 1
        self.category_sep = ' :: '
        self.date_sep = ' – '
        self.first_and_last = 0
        self.horz = 1
        self.included_cats = []
        self.min_start_time = '02:00:00'
        self.pairwise = 1


def main():
    # _params = {
    #     'horz': 1,
    #     'categories_out': 1,
    #     'categories': 1,
    #     'category_sep': ' :: ',
    #     'date_sep': ' – ',
    #     'pairwise': 1,
    #     'included_cats': 0,
    #     'first_and_last': 0,
    #     'add_date': 1,
    #     'add_diff': 1,
    #     'add_comment': 1,
    #     'min_start_time': '03:00:00',
    # }

    _params = Params()
    paramparse.process(_params)

    horz = _params.horz
    categories_out = _params.categories_out
    categories = _params.categories
    category_sep = _params.category_sep
    date_sep = _params.date_sep
    first_and_last = _params.first_and_last
    pairwise = _params.pairwise
    add_date = _params.add_date
    add_diff = _params.add_diff
    add_comment = _params.add_comment
    min_start_time = _params.min_start_time
    included_cats = _params.included_cats

    try:
        in_txt = Tk().clipboard_get()  # type: str
    except BaseException as e:
        print('Tk().clipboard_get() failed: {}'.format(e))
        return

    lines = [k.strip() for k in in_txt.split('\n') if k.strip()]

    min_start_time_obj = datetime.strptime(min_start_time, '%H:%M:%S').time()
    midnight_time_obj = datetime.strptime('00:00:00', '%H:%M:%S').time()

    out_lines = []
    out_times = []
    out_date_times = []
    # out_date_time_str = []
    out_categories = []

    date_str = datetime.now().strftime("%Y-%m-%d")
    curr_comments = []

    out_comments = []

    for line in lines:
        category = None
        if categories:
            if category_sep in line:
                temp = line.split(category_sep)

                # print('line: {}'.format(line))
                # print('temp: {}'.format(temp))

                if len(temp) == 2:
                    line, category = temp
                    line = line.strip()
                    category = category.strip()

        if date_sep in line:
            temp = line.split(date_sep)

            _, time_found, _ = is_time(temp[0])
            if time_found:
                print('date_sep line: {}'.format(line))
                print('date_sep temp: {}'.format(temp))

                if len(temp) == 3:
                    line, _, date_str = temp
                    line = line.strip()
                elif len(temp) == 2:
                    line, possible_date = temp
                    line = line.strip()
                    _, time_found, _ = is_time(possible_date)
                    if not time_found:
                        date_str = possible_date

        line, time_found, time_obj = is_time(line)

        if time_found:
            curr_comments_str = ''

            if curr_comments:
                curr_comments_str = ';'.join(curr_comments)
                curr_comments = []

            #     if not out_lines:
            #         print('dangling comment found: {}'.format(curr_comments_str))
            # else:
            #     if out_lines:
            #         print('no comment found for: {}'.format(line))

            out_comments.append(curr_comments_str)

            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()

            if midnight_time_obj <= time_obj < min_start_time_obj:
                date_obj = date_obj + timedelta(days=1)

            date_time_obj = datetime.combine(date_obj, time_obj)

            # date_time_str = date_time_obj.strftime("%Y%m%d_%H%M%S")
            # out_date_time_str.append(date_time_str)

            out_date_times.append(date_time_obj)
            out_times.append(time_obj)
            out_lines.append(line)

            if categories:
                if category is None:
                    category = -1
                out_categories.append(category)
        else:
            curr_comments.append(line)

    if curr_comments:
        curr_comments_str = ';'.join(curr_comments)
    else:
        curr_comments_str = ''

    out_comments.append(curr_comments_str)

    sort_ids = [i[0] for i in sorted(enumerate(out_date_times), key=lambda x: x[1])]

    # out_date_times = [out_date_times[i] for i in sort_ids]
    # # out_date_time_str = [out_date_time_str[i] for i in sort_ids]
    # out_times = [out_times[i] for i in sort_ids]
    # out_lines = [out_lines[i] for i in sort_ids]
    # out_comments = [out_comments[i] for i in sort_ids]

    if first_and_last and len(out_lines) > 2:
        out_date_times = [out_date_times[sort_ids[0]], out_date_times[sort_ids[-1]]]
        out_lines = [out_lines[sort_ids[0]], out_lines[sort_ids[-1]]]

    if included_cats:
        included_cats = [str(k) for k in included_cats]
        print(f'including only categories: {included_cats}')

    n_out_lines = len(out_lines)
    if pairwise:
        out_txt0 = ''
        out_txt = ''
        out_txt2 = ''
        out_txt3 = ''
        for _line_id in range(n_out_lines - 1):

            curr_sort_id, next_sort_id = sort_ids[_line_id], sort_ids[_line_id + 1]

            if categories and included_cats and out_categories[next_sort_id] not in included_cats:
                print(f'skipping entry with category {out_categories[next_sort_id]}')
                continue

            if horz:
                _out_txt = '{}\t{}'.format(out_lines[curr_sort_id], out_lines[next_sort_id])

                if add_date:
                    _out_txt = '{}\t{}'.format(date_str, _out_txt)
                if categories_out:
                    _out_txt += '\t{}'.format(out_categories[next_sort_id])

                if add_diff:
                    time_diff = out_date_times[next_sort_id] - out_date_times[curr_sort_id]
                    time_diff_str = str(time_diff)

                    if ',' in time_diff_str:
                        """times from different days across midnight"""
                        print('times from different days across midnight found')
                        input()
                        exit()
                        # time_diff_str = time_diff_str.split(',')[-1].strip()

                    _out_txt += '\t{}'.format(time_diff_str)

                if add_comment:
                    """out_comments has an annoying extra entry at top"""
                    _out_txt = '{}\t{}'.format(_out_txt, out_comments[next_sort_id + 1])

                out_txt += _out_txt + '\n'
            else:
                out_txt += '{}\t'.format(out_lines[curr_sort_id])
                out_txt2 += '{}\t'.format(out_lines[next_sort_id])
                if add_date:
                    out_txt0 = '{}\t'.format(date_str)
                if categories_out:
                    out_txt3 += '{}\t'.format(out_categories[next_sort_id])
        if not horz:
            out_txt += '\n' + out_txt2
            if add_date:
                out_txt = '{}\n{}'.format(out_txt0, out_txt)
            if categories:
                out_txt += '\n' + out_txt3
    else:
        if horz:
            field_sep = '\t'
        else:
            field_sep = '\n'

        out_txt = field_sep.join(out_lines)

    out_txt = out_txt.rstrip()

    print('out_txt:\n{}'.format(out_txt))

    try:
        import pyperclip

        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


if __name__ == '__main__':
    main()
