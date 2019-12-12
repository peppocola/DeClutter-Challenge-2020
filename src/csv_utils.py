from pandas import read_csv
import csv
from src.keys import key_classifier, key_metric
import matplotlib.pyplot as plt
import numpy as np

datapath = '../csv/declutter-gold_DevelopmentSet.csv'
csv_outpath = '../csv/'
img_outpath = '../img/'


def write_counter(counter):
    nameto = 'count.csv'
    with open(csv_outpath + nameto, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "Yes", "No", "NI Rate"])
        for key in counter:
            writer.writerow([key, counter[key][0], counter[key][1],
                             round(counter[key][0] / (counter[key][0] + counter[key][1]), 2)])


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
    counter = {}
    counter['Javadoc'] = [0, 0]
    counter['Line'] = [0, 0]
    counter['Block'] = [0, 0]

    for i in range(lines.__len__()):
        if lines.iloc[i]['non-information'] == 'yes':
            counter[lines.iloc[i]['type']][1] += 1
        else:
            counter[lines.iloc[i]['type']][0] += 1
    return counter


def commentparser():
    lines = read_csv(datapath,
                     sep=",", usecols=['comment'])
    return lines.comment.fillna(' ').tolist()


def labelparser():
    lines = read_csv(datapath,
                     sep=",", usecols=['non-information'])
    return [0 if x == 'no' else 1 for x in lines['non-information'].tolist()]


def plot_length():
    comments = np.array([len(x) for x in commentparser()])
    labels = np.array(labelparser())

    sample_ni = []
    sample_nni = []
    for x in range(len(comments)):
        if labels[x] == 0:
            sample_ni.append(comments[x])
        else:
            sample_nni.append(comments[x])

    plt.hist(sample_ni, bins='auto', color='blue')
    plt.hist(sample_nni, bins='auto', color='orange')
    plt.xlabel('comment length')
    plt.ylabel('number of comments')
    plt.text(1000, 110, 'ni avg length= ' + str(round(sum(sample_ni) / len(sample_ni), 2)))
    plt.text(1000, 100, 'nni avg length=' + str(round(sum(sample_nni) / len(sample_nni), 2)))
    plt.savefig(img_outpath + 'length_distribution.png')


if __name__ == "__main__":
    write_counter(csv_counter())
    print(labelparser())
    print(commentparser())
    plot_length()
