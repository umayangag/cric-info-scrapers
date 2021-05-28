import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data\\bowling_data_cluster.csv')

bowling_performance_data = pd.read_csv(filename)

print(bowling_performance_data)
print(bowling_performance_data.isna().sum())


# distortions = []
# K = range(1, 10)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k)
#     kmeanModel.fit(bowling_performance_data)
#     distortions.append(kmeanModel.inertia_)
#
# plt.figure(figsize=(16, 8))
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k for bowling performance')
# plt.show()


def cluster_bowling_performance(dataset):
    kmeanModel = KMeans(n_clusters=3)
    kmeanModel.fit(dataset)

    dataset['bowling_performance'] = kmeanModel.predict(dataset)
    return dataset


# cluster_bowling_performance(bowling_performance_data)
#
# output_file = os.path.join(dirname, 'output\\bowling_cluster.csv')
# bowling_performance_data.to_csv(output_file)
#
# x = "wickets"
# y = "runs"
# plt.scatter(bowling_performance_data[x], bowling_performance_data[y],
#             c=bowling_performance_data['bowling_performance'])
# plt.title("cluster visualization")
# plt.xlabel(x)
# plt.ylabel(y)
# plt.show()
