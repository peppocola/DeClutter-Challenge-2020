import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.keys import non_information, img_outpath
from src.csv_utils import commentparser, labelparser


# takes a confusion matrix and plots it
def saveHeatmap(cm, name):
    sns.heatmap(cm, square=False, annot=True, cbar=True, fmt="d", xticklabels=["information", "non_information"],
                yticklabels=["information", "non_information"])
    plt.text(-0.25, 1.15, 'actual', rotation=90)
    plt.text(0.8, 2.2, 'predicted')
    plt.savefig(img_outpath + 'heatmap_' + name)
    plt.clf()


def plot_length():
    comments = np.array([len(x) for x in commentparser()])
    labels = np.array(labelparser())

    sample_ni = []
    sample_nni = []
    for x in range(len(comments)):
        if labels[x] == non_information:
            sample_ni.append(comments[x])
        else:
            sample_nni.append(comments[x])

    plt.hist(sample_nni, bins='auto', color='orange')
    plt.hist(sample_ni, bins='auto', color='blue')

    plt.xlabel('comment length')
    plt.ylabel('number of comments')
    plt.legend(['non-information', 'non non-information'])
    plt.text(1000, 90, 'ni avg length= ' + str(round(sum(sample_ni) / len(sample_ni), 2)))
    plt.text(1000, 80, 'nni avg length=' + str(round(sum(sample_nni) / len(sample_nni), 2)))
    plt.savefig(img_outpath + 'length_distribution.png')
    plt.clf()


if __name__ == "__main__":
    plot_length()
