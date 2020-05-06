import os

from pandas import read_csv, DataFrame
import csv

from sklearn.model_selection import train_test_split

from src.keys import non_information, information, full_train_path, reports_outpath, scores_outpath, \
    csv_ex, java_tags, java_keywords, javadoc, features_outpath, split_test_path, latex_tables_out, split_train_path, \
    new_train_path, new_test_path


def write_counter(counter):
    nameto = 'count' + csv_ex
    with open(reports_outpath + nameto, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "Yes", "No", "NI Rate"])
        for key in counter:
            writer.writerow([key, counter[key][non_information], counter[key][information],
                             round(counter[key][non_information] / (
                                     counter[key][information] + counter[key][non_information]), 2)])


def write_stats(stats, folder):
    name_full = scores_outpath + folder + "/stats" + csv_ex

    if not os.path.exists(scores_outpath + folder):
        os.makedirs(scores_outpath + folder)

    rows = []
    header = []
    for key in stats:
        desc_row, values = write_classifier_stats(stats[key], key, folder)  # write single scores
        header = desc_row
        rows.append(values)
    header.insert(0, "classifier")

    with open(name_full, mode='w', newline='') as file:  # write full scores (cat the single)
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    write_latex_tables(name_full, latex_tables_out, folder)


def write_latex_tables(name_full, latex_tables_out, folder):
    name_latex_class = latex_tables_out + folder + "/table_yes_no" + csv_ex
    name_latex_short = latex_tables_out + folder + "/table_short_" + csv_ex

    if not os.path.exists(latex_tables_out + folder):
        os.makedirs(latex_tables_out + folder)

    f = read_csv(name_full)

    latex_class_cols = ['classifier', 'precision_no', 'recall_no', 'f1_no', 'precision_yes', 'recall_yes',
                        'f1_yes']

    latex_class = f[latex_class_cols]
    latex_class.to_csv(name_latex_class, index=False)

    latex_short_cols = ['classifier', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                        'matthews_corrcoef']

    latex_short = f[latex_short_cols]
    latex_short.to_csv(name_latex_short, index=False)


def write_classifier_stats(stats, key, folder):
    path = scores_outpath + folder + '/' + key + csv_ex
    desc_row = []
    value_row = []
    for k in stats:
        if type(stats[k]) is dict:
            for h in stats[k]:
                desc_row.append(h + '_' + k)
                value_row.append(round(stats[k][h], 2))
        else:
            desc_row.append(k)
            value_row.append(round(stats[k], 2))

    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(desc_row)
        writer.writerow(value_row)

    value_row.insert(0, key)
    return desc_row, value_row


def write_csv(col_names, data, filename):
    with open(features_outpath + '\\' + filename + csv_ex, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(col_names)
        for i in range(len(data[0])):
            row = []
            for values in data:
                row.append(values[i])
            writer.writerow(row)


def csv_counter():
    lines = read_csv(full_train_path,
                     sep=",", usecols=['type', 'non-information'])
    counter = {'Javadoc': [0, 0], 'Line': [0, 0], 'Block': [0, 0]}

    for i in range(lines.__len__()):
        if lines.iloc[i]['non-information'] == 'yes':
            counter[lines.iloc[i]['type']][non_information] += 1
        else:
            counter[lines.iloc[i]['type']][information] += 1
    return counter


def get_comments(set='train'):
    path = get_path(set)
    lines = read_csv(path, sep=",", usecols=['comment'])
    return lines.comment.fillna(' ').tolist()


def get_type(set='train'):
    path = get_path(set)
    lines = read_csv(path, sep=",", usecols=['type'])
    return lines.type.fillna(' ').tolist()


def get_labels(set='train'):
    path = get_path(set)

    lines = read_csv(path, sep=",", usecols=['non-information'])
    return [information if x == 'no' else non_information for x in lines['non-information'].tolist()]


def get_links():
    lines = read_csv(full_train_path,
                     sep=",", usecols=['path_to_file'])
    return lines['path_to_file'].tolist()


def get_tags():
    with open(java_tags, 'r') as f:
        return [line for line in f.read().splitlines()]


def get_keywords():
    with open(java_keywords, 'r') as f:
        return [keyword for keyword in f.read().splitlines()]


def get_link_line_type(set='train'):
    path = get_path(set)
    lines = read_csv(path, sep=",", usecols=['type', 'path_to_file', 'begin_line'])
    return lines.values.tolist()


def get_javadoc_comments():
    comments = get_comments()
    types = get_type()

    javadoc_comments = []
    for i in range(len(comments)):
        if types[i] == javadoc:
            javadoc_comments.append(comments[i])

    return javadoc_comments


def write_results(results):
    df = read_csv(split_test_path, sep=",",
                  usecols=['ID'])
    ids = df.ID.fillna(' ').tolist()
    non_information_col = [0 if x == 0 else 1 for x in results]
    out = DataFrame()
    out['ID'] = ids
    out['Expected'] = non_information_col
    out.to_csv('../devset/out.csv', index=False)


def get_path(set_name='train'):
    if set_name == 'train':
        path = full_train_path
    elif set_name == 'split_test':
        path = split_test_path
    elif set_name == 'split_train':
        path = split_train_path
    elif set_name == 'new_train':
        path = new_train_path
    elif set_name == 'new_test':
        path = new_test_path
    else:
        raise ValueError
    return path


def data_split():
    df = read_csv(full_train_path, sep=",")
    split = train_test_split(df, shuffle=True, test_size=0.1, train_size=0.9)
    train = split[0]
    test = split[1]
    train.to_csv(new_train_path, index=False)
    test.to_csv(new_test_path, index=False)
    return new_train_path, new_test_path


def fill_test_set_labels():
    to_fill = read_csv(split_test_path, sep=",")
    link_labels = get_link_label_dict()
    for index, row in to_fill.iterrows():
        to_fill['non-information'][index] = link_labels[to_fill['link_to_comment'][index]]
    to_fill.to_csv(split_test_path, index=False)


def get_link_label_dict():
    link_label = read_csv(full_train_path, sep=",", usecols=['link_to_comment', 'non-information'])
    link_label = link_label.values.tolist()
    link_label_dict = {}
    for entry in link_label:
        link_label_dict[entry[0]] = entry[1]
    return link_label_dict


if __name__ == "__main__":
    data_split()
    # write_counter(csv_counter())
    # print(get_labels())
    # print(get_comments())
    # print(get_links())
    #print(get_link_line_type())
    #print(get_link_label_dict())
    fill_test_set_labels()