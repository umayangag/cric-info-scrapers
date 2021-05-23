from sklearn import preprocessing
import pandas as pd

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data\\batting_data_performance.csv')
batting_performance_data = pd.read_csv(filename)

# normalize the data attributes
normalized = preprocessing.normalize(batting_performance_data)
print(normalized)
# output_file = os.path.join(dirname, 'data\\batting_data_performance_normalized.csv')
# batting_performance_data.to_csv(output_file)