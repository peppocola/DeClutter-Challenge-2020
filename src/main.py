from src.classifier import classify
from src.csv_utils import write_counter, csv_counter, write_stats
from src.plot_utils import plot_length
import time

write_counter(csv_counter())
plot_length()
start_time = time.time()
write_stats(classify())
print("--- %s seconds ---" % (time.time() - start_time))
