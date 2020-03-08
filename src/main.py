from src.classifier import classify
from src.csv_utils import write_counter, csv_counter, write_stats
from src.plot_utils import plot_length, tags_analysis, has_tags_analysis
import time

write_counter(csv_counter())
plot_length()
start_time = time.time()
write_stats(classify())
has_tags_analysis()
tags_analysis()

print("--- %s seconds ---" % (time.time() - start_time))
