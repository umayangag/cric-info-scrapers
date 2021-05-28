import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os

find_k = False


def cluster_batting_performance(dataset):
    kmeanModel = KMeans(n_clusters=3)
    kmeanModel.fit(dataset)

    dataset['batting_performance'] = kmeanModel.predict(dataset)
    return dataset


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    # filename = os.path.join(dirname, 'cluster_data\\batting_performance.csv')
    filename = os.path.join(dirname, 'data\\batting_data_cluster.csv')

    batting_performance_data = pd.read_csv(filename)

    print(batting_performance_data)

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
        cluster_batting_performance(batting_performance_data)

        output_file = os.path.join(dirname, 'output\\batting_cluster.csv')
        batting_performance_data.to_csv(output_file)
        x = "strike_rate"
        y = "runs"

        plt.scatter(batting_performance_data[x], batting_performance_data[y],
                    c=batting_performance_data['batting_performance'])
        plt.title("cluster visualization")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
