from src.classifier import classify, feat_classify
from src.csv_utils import write_counter, csv_counter
from src.plot_utils import plot_length, tags_analysis, has_tags_analysis, plot_jaccard

write_counter(csv_counter())
plot_length()
plot_jaccard()
plot_jaccard(stemming=False, rem_kws=False)
classify()
feat_classify()
has_tags_analysis()
tags_analysis()

