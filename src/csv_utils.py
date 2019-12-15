from pandas import read_csv
import csv
from src.keys import key_classifier, key_metric, non_information, information, datapath, csv_outpath


def write_counter(counter):
    nameto = 'count.csv'
    with open(csv_outpath + nameto, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "Yes", "No", "NI Rate"])
        for key in counter:
            writer.writerow([key, counter[key][non_information], counter[key][information],
                             round(counter[key][non_information] / (
                                         counter[key][information] + counter[key][non_information]), 2)])


def write_stats(stats):
    nameto = 'stats.csv'
    with open(csv_outpath + nameto, mode='w', newline='') as file:
        writer = csv.writer(file)
        keys = [x for x in stats[key_metric]]
        writer.writerow([x for x in ['classifier'] + keys])
        for classifier in stats[key_classifier]:
            writer.writerow([classifier] + stats[classifier])


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


if __name__ == "__main__":
    write_counter(csv_counter())
    print(labelparser())
    print(commentparser())
