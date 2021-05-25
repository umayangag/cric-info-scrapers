from sklearn import preprocessing
import pandas as pd

import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data\\bowling_data_performance.csv')
# bowling_performance_data = pd.read_csv(filename)


# normalize the data attributes
def normalize_bowling_dataset(dataset):
    numeric_columns = ['temp', 'wind', 'rain', 'humidity', 'cloud', 'pressure', 'venue', 'opposition',
                       'player_consistency', 'player_form', "inning", 'viscosity', 'season']
    normalized = preprocessing.normalize(dataset[numeric_columns])
    df_normalized = pd.DataFrame(normalized, columns=numeric_columns)
    for column in numeric_columns:
        dataset[column] = df_normalized[column]
    return dataset


# normalize_bowling_dataset(bowling_performance_data)
# print(bowling_performance_data)
# output_file = os.path.join(dirname, 'data\\bowling_data_performance_normalized.csv')
# bowling_performance_data.to_csv(output_file)
