import pandas as pd
import csv

path = 'declutter-gold_DevelopmentSet.csv'


def writeOut(counter):
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "Yes", "No", "NI Rate"])
        for key in counter:
            writer.writerow([key, counter[key][0], counter[key][1],
                             round(counter[key][0] / (counter[key][0] + counter[key][1]), 2)])


def csv_parser():
    lines = pd.read_csv(path,
                        sep=",", usecols=['type', 'non-information'])
    counter = {}
    counter['Javadoc'] = [0, 0]
    counter['Line'] = [0, 0]
    counter['Block'] = [0, 0]

    for i in range(lines.__len__()):
        if lines.iloc[i]['non-information'] == 'yes':
            counter[lines.iloc[i]['type']][0] += 1
        else:
            counter[lines.iloc[i]['type']][1] += 1
    return counter


def commentparser():
    lines = pd.read_csv(path,
                        sep=",", usecols=['comment'])
    return lines.comment.fillna(' ').tolist()


def labelparser():
    lines = pd.read_csv(path,
                        sep=",", usecols=['non-information'])
    return [0 if x == 'no' else 1 for x in lines['non-information'].tolist()]


if __name__ == "__main__":
    writeOut(csv_parser())
    print(labelparser())
    print(commentparser())
