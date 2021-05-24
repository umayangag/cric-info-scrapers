import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'cluster_data\\bowling_performance.csv')

bowling_performance_data = pd.read_csv(filename)

print(bowling_performance_data)
print(bowling_performance_data.isna().sum())

# distortions = []
# K = range(1,10)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k)
#     kmeanModel.fit(bowling_performance_data)
#     distortions.append(kmeanModel.inertia_)

# plt.figure(figsize=(16,8))
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k for bowling performance')
# plt.show()

kmeanModel = KMeans(n_clusters=3)
kmeanModel.fit(bowling_performance_data)

bowling_performance_data['bowling_performance']=kmeanModel.predict(bowling_performance_data)
output_file = os.path.join(dirname, 'output\\bowling_cluster.csv')
bowling_performance_data.to_csv(output_file)

plt.scatter(bowling_performance_data["wickets"], bowling_performance_data["econ"], c=bowling_performance_data['bowling_performance'])
plt.title("cluster visualization")
plt.xlabel('wickets')
plt.ylabel('econ')
plt.show()