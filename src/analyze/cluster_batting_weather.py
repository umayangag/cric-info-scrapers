import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os

find_k = True
find_k = False

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    filename_batting = os.path.join(dirname, 'cluster_data\\batting_encoded.csv')

    batting_performance_data = pd.read_csv(filename_batting)

    if find_k:

        distortions = []
        K = range(1, 10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(batting_performance_data)
            distortions.append(kmeanModel.inertia_)

        plt.figure(figsize=(16, 8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k for batting performance')
        plt.show()

    else:

        kmeanModel = KMeans(n_clusters=5)
        kmeanModel.fit(batting_performance_data)

        batting_performance_data['weather'] = kmeanModel.predict(batting_performance_data)
        output_file = os.path.join(dirname, 'output\\batting_weather.csv')
        batting_performance_data.to_csv(output_file)

        # humidity, cloud, pressure
        x = "runs_scored"
        y = "batting_cloud"

        plt.scatter(batting_performance_data[x], batting_performance_data[y], c=batting_performance_data['weather'])
        plt.title("cluster visualization")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
