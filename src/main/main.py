import json

from src.classification.classification import classify_split, tf_idf_classify, feat_classify, both_classify
from src.comment_analysis.parsing_utils import get_lines

selected_for_voting = classify_split()
stats, voting = tf_idf_classify()
selected_for_voting.append(voting)
print("getting relevant lines")
lines = get_lines(serialized=True)
print("features\n")
stats, voting = feat_classify(lines=lines)
selected_for_voting.append(voting)
print("both\n")
stats, voting = both_classify(lines=lines)

selected_for_voting.append(voting)
x = open('../serialized_' + "voting" + '.json', 'w')
x.write(json.dumps(selected_for_voting))