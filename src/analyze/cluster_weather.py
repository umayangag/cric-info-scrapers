import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)
filename_batting = os.path.join(dirname, 'cluster_data\\batting-weather.csv')
filename_bowling = os.path.join(dirname, 'cluster_data\\bowling-weather.csv')

batting_performance_data = pd.read_csv(filename_batting)
bowling_performance_data = pd.read_csv(filename_bowling)

print(batting_performance_data)
print(bowling_performance_data)

df1 = pd.concat([batting_performance_data, bowling_performance_data], ignore_index=True)
print(df1)

distortions = []
K = range(1, 10)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k)
#     kmeanModel.fit(df1)
#     distortions.append(kmeanModel.inertia_)

# plt.figure(figsize=(16,8))
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k for batting performance')
# plt.show()
#
kmeanModel = KMeans(n_clusters=4)
kmeanModel.fit(df1)

df1['weather'] = kmeanModel.predict(df1)
# output_file = os.path.join(dirname, 'output\\batting_weather.csv')
# batting_performance_data.to_csv(output_file)

# for i in range(0, 7):
#     for j in range(0, 7):
#         x = df1.columns[i]
#         y = df1.columns[j]
#
#         if x < y:
#             plt.scatter(df1[x], df1[y], c=df1['weather'])
#             plt.title("cluster visualization")
#             plt.xlabel(x)
#             plt.ylabel(y)
#             plt.savefig(os.path.join(dirname, f'graphs\\scatter_{x}_vs_{y}.png'), format="png")

# humidity, cloud, pressure
x = "cloud"
y = "pressure"

plt.scatter(df1[x], df1[y], c=df1['weather'])
plt.title("cluster visualization")
plt.xlabel(x)
plt.ylabel(y)
plt.show()
