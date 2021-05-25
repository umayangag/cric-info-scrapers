import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'cluster_data\\batting_performance.csv')

batting_performance_data = pd.read_csv(filename)

print(batting_performance_data)
print(batting_performance_data.isna().sum())


# distortions = []
# K = range(1,10)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k)
#     kmeanModel.fit(batting_performance_data)
#     distortions.append(kmeanModel.inertia_)

# plt.figure(figsize=(16,8))
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k for batting performance')
# plt.show()

def classify_batting_performance(dataset):
    kmeanModel = KMeans(n_clusters=3)
    kmeanModel.fit(dataset)

    dataset['batting_performance'] = kmeanModel.predict(dataset)
    return dataset


# classify_batting_performance(batting_performance_data)
#
# output_file = os.path.join(dirname, 'output\\batting_cluster.csv')
# batting_performance_data.to_csv(output_file)
#
# plt.scatter(batting_performance_data["runs"], batting_performance_data["performance"],
#             c=batting_performance_data['batting_performance'])
# plt.title("cluster visualization")
# plt.xlabel('strike rate')
# plt.ylabel('runs scored')
# plt.show()
