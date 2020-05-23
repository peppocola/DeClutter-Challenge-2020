from src.classification.kaggle_classifier import kaggle_classify
from src.csv.csv_utils import write_results

if __name__ == "__main__":
    result, set_name = kaggle_classify()
    write_results(result, set_name)
