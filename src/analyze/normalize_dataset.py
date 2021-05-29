from sklearn import preprocessing
import pandas as pd

import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data\\batting_data_performance.csv')


# batting_performance_data = pd.read_csv(filename)


# normalize the data attributes
def normalize_dataset(dataset):
    numeric_columns = [
        "contribution",
        "total_score",
        "total_balls",
        "player_consistency",
        "player_form",
        "runs",
        "balls",
        "fours",
        "sixes",
        "strike_rate",
        "temp",
        "wind",
        "rain",
        "humidity",
        "cloud",
        "pressure",
        "venue",
        "opposition",
    ]
    normalized = preprocessing.normalize(dataset[numeric_columns])
    df_normalized = pd.DataFrame(normalized, columns=numeric_columns)
    for column in numeric_columns:
        dataset[column] = df_normalized[column]
    return dataset