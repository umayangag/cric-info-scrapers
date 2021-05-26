from sklearn import preprocessing
import pandas as pd

import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data\\batting_data_performance.csv')


# batting_performance_data = pd.read_csv(filename)


# normalize the data attributes
def normalize_batting_dataset(dataset):
    numeric_columns = ['temp', 'wind', 'rain', 'humidity', 'cloud', 'pressure', 'venue', 'opposition',
                       'player_consistency', 'player_form', 'viscosity']
    normalized = preprocessing.normalize(dataset[numeric_columns])
    df_normalized = pd.DataFrame(normalized, columns=numeric_columns)
    for column in numeric_columns:
        dataset[column] = df_normalized[column]
    return dataset

# normalize_batting_dataset(batting_performance_data)
# print(batting_performance_data)
# output_file = os.path.join(dirname, 'data\\batting_data_performance_normalized.csv')
# batting_performance_data.to_csv(output_file)
