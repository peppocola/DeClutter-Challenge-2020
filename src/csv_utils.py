from pandas import read_csv
import csv
from src.keys import non_information, information, datapath, reports_outpath, scores_outpath, \
    csv_ex, java_tags


def write_counter(counter):
    nameto = 'count'+ csv_ex
    with open(reports_outpath + nameto, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "Yes", "No", "NI Rate"])
        for key in counter:
            writer.writerow([key, counter[key][non_information], counter[key][information],
                             round(counter[key][non_information] / (
                                     counter[key][information] + counter[key][non_information]), 2)])


def write_stats(stats):
    name_full = scores_outpath + "full_stats" + csv_ex
    name_short = scores_outpath + "short_stats" + csv_ex

    rows = []
    header = []
    for key in stats:
        desc_row, values = write_classifier_stats(stats[key], key)  # write single scores
        header = desc_row
        rows.append(values)
    header.insert(0, "classifier")

    with open(name_full, mode='w', newline='') as file:  # write full scores (cat the single)
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    f = read_csv(name_full)
    keep_col = ['classifier', 'precision_no', 'recall_no', 'f1-score_no', 'precision_yes', 'recall_yes', 'f1-score_yes',
                # keep some cols
                'accuracy', 'precision_macro avg', 'recall_macro avg', 'f1-score_macro avg']
    new_f = f[keep_col]
    new_f.to_csv(name_short, index=False)


def write_classifier_stats(stats, key):
    path = scores_outpath + key + csv_ex
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


def csv_counter():
    lines = read_csv(datapath,
                     sep=",", usecols=['type', 'non-information'])
    counter = {'Javadoc': [0, 0], 'Line': [0, 0], 'Block': [0, 0]}

    for i in range(lines.__len__()):
        if lines.iloc[i]['non-information'] == 'yes':
            counter[lines.iloc[i]['type']][non_information] += 1
        else:
            counter[lines.iloc[i]['type']][information] += 1
    return counter


def commentparser():
    lines = read_csv(datapath,
                     sep=",", usecols=['comment'])
    return lines.comment.fillna(' ').tolist()


def labelparser():
    lines = read_csv(datapath,
                     sep=",", usecols=['non-information'])
    return [information if x == 'no' else non_information for x in lines['non-information'].tolist()]


def linkparser():
    lines = read_csv(datapath,
                     sep=",", usecols=['path_to_file'])
    return lines['path_to_file'].tolist()


def tagsparser():
    with open(java_tags, 'r') as f:
        return [line for line in f.read().splitlines()]

if __name__ == "__main__":
    write_counter(csv_counter())
    print(labelparser())
    print(commentparser())
    print(linkparser())
