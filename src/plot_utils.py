import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.keys import non_information, information, img_outpath, reports_outpath
from src.csv_utils import get_comments, get_labels, get_tags
import re
from src.feature_extractor import jaccard


def save_heatmap(cm, name, folder):
    sns.heatmap(cm, square=False, annot=True, cbar=True, fmt="d", xticklabels=["no", "yes"],
                yticklabels=["no", "yes"])
    plt.text(-0.25, 1.15, 'predicted', rotation=90)
    plt.text(0.8, 2.2, 'actual')
    plt.savefig(img_outpath + folder + '/heatmap_' + name)
    plt.clf()


def plot_length():
    comments = np.array([len(x) for x in get_comments()])
    labels = np.array(get_labels())

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
    plt.legend(['no', 'yes'])
    plt.text(1000, 80, 'yes avg length= ' + str(round(sum(sample_ni) / len(sample_ni), 2)))
    plt.text(1000, 90, 'no avg length=' + str(round(sum(sample_nni) / len(sample_nni), 2)))
    plt.savefig(img_outpath + 'length_distribution.png')
    plt.clf()


def has_tags_analysis():
    has_tags = 1
    no_tags = 0
    tags = get_tags()
    comments = get_comments()
    labels = np.array(get_labels())

    tag_comment = []
    for comment in comments:
        match = False
        for tag in tags:
            if re.search(tag, comment):
                match = True
                tag_comment.append(has_tags)
                break
        if match is False:
            tag_comment.append(no_tags)

    tags_positive = 0
    tags_negative = 0
    no_tags_positive = 0
    no_tags_negative = 0

    for i in range(len(labels)):
        if labels[i] == non_information and tag_comment[i] == has_tags:
            tags_positive += 1
        elif labels[i] == information and tag_comment[i] == has_tags:
            tags_negative += 1
        elif labels[i] == non_information and tag_comment[i] == no_tags:
            no_tags_positive += 1
        elif labels[i] == information and tag_comment[i] == no_tags:
            no_tags_negative += 1
        i += 1
    with open(reports_outpath + "has_tags" + ".txt", 'w') as f:
        f.write("yes w tags = " + str(tags_positive) + "/" + str(tags_positive + no_tags_positive) + "\n")
        f.write("yes wout tags = " + str(no_tags_positive) + "/" + str(tags_positive + no_tags_positive) + "\n")
        f.write("no w tags = " + str(tags_negative) + "/" + str(tags_negative + no_tags_negative) + "\n")
        f.write("no wout tags = " + str(no_tags_negative) + "/" + str(tags_negative + no_tags_negative) + "\n")
    assert tags_positive + tags_negative + no_tags_positive + no_tags_negative == len(labels)


def tags_analysis():
    labels = np.array(get_labels())
    comments = get_comments()
    tags = get_tags()

    tags_dict = {}
    for tag in tags:
        tags_dict[tag] = [0, 0]
        i = 0
        for comment in comments:
            if re.search(tag, comment):
                if labels[i] == non_information:
                    tags_dict[tag][non_information] += 1
                else:
                    tags_dict[tag][information] += 1
            i += 1
    with open(reports_outpath + "tags_analysis" + ".txt", 'w') as f:
        for key in tags_dict:
            if tags_dict[key] != [0, 0]:
                f.write(key + ":" + "\n")
                f.write("\tno -> " + str(tags_dict[key][information]) + "\n")
                f.write("\tyes-> " + str(tags_dict[key][non_information]) + "\n")


def plot_jaccard(stemming=True, rem_kws=True):
    jaccard_scores = np.array(jaccard(stemming, rem_kws))
    labels = np.array(get_labels())
    img_ext = '.png'
    outpath_yes = img_outpath + 'jacc_distribution_yes'
    outpath_no = img_outpath + 'jacc_distribution_no'
    if stemming:
        outpath_yes += '_stem_'
        outpath_no += '_stem'
    if rem_kws:
        outpath_yes += '_remkws'
        outpath_no += '_remkws'
    outpath_yes += img_ext
    outpath_no += img_ext

    sample_ni = []
    sample_nni = []
    for x in range(len(labels)):
        if labels[x] == non_information:
            sample_ni.append(jaccard_scores[x])
        else:
            sample_nni.append(jaccard_scores[x])

    plt.hist(sample_ni, bins='auto', color='orange')

    plt.xlabel('jacc_score')
    plt.ylabel('number of comments')
    plt.ylim(0, 200)
    plt.xlim(0, 1)
    plt.legend(['yes'])
    plt.text(0.7, 150, 'yes avg jacc=' + str(round(sum(sample_nni) / len(sample_nni), 3)))
    plt.savefig(outpath_yes)
    plt.clf()

    plt.hist(sample_nni, bins='auto', color='blue')

    plt.xlabel('jacc_score')
    plt.ylabel('number of comments')
    plt.ylim(0, 200)
    plt.xlim(0, 1)
    plt.legend(['no'])
    plt.text(0.7, 150, 'no avg jacc= ' + str(round(sum(sample_ni) / len(sample_ni), 3)))
    plt.savefig(outpath_no)
    plt.clf()


if __name__ == "__main__":
    #plot_length()
    #has_tags_analysis()
    #tags_analysis()
    #plot_jaccard(stemming=False, rem_kws=False)
    plot_jaccard(stemming=False, rem_kws=False)
