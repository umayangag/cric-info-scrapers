import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'cluster_data\\batting-weather.csv')

batting_performance_data = pd.read_csv(filename)

print(batting_performance_data)
print(batting_performance_data.isna().sum())

df1 = batting_performance_data.loc[:, batting_performance_data.columns != "performance"]
print(df1)

# distortions = []
# K = range(1,10)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k)
#     kmeanModel.fit(df1)
#     distortions.append(kmeanModel.inertia_)
#
# plt.figure(figsize=(16,8))
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k for batting performance')
# plt.show()

kmeanModel = KMeans(n_clusters=4)
kmeanModel.fit(df1)

batting_performance_data['weather']=kmeanModel.predict(df1)
# output_file = os.path.join(dirname, 'output\\batting_weather.csv')
# batting_performance_data.to_csv(output_file)

plt.scatter(batting_performance_data["temp"], batting_performance_data["humidity"], c=batting_performance_data['performance'])
plt.title("cluster visualization")
plt.xlabel('temperature')
plt.ylabel('humidity')
plt.show()
