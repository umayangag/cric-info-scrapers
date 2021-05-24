from sklearn import preprocessing
import pandas as pd

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data\\bowling_data_performance.csv')
bowling_performance_data = pd.read_csv(filename)

# normalize the data attributes
numeric_columns=['temp','wind','rain','humidity','cloud','pressure']
normalized = preprocessing.normalize(bowling_performance_data[numeric_columns])
df_normalized= pd.DataFrame(normalized, columns=numeric_columns)
for column in numeric_columns:
    bowling_performance_data[column]=df_normalized[column]
print(bowling_performance_data)
output_file = os.path.join(dirname, 'data\\bowling_data_performance_normalized.csv')
bowling_performance_data.to_csv(output_file)  