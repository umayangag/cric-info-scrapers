import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans

from config.mysql import get_db_connection
from final_data.encoders import encode_viscosity

find_k = True
# find_k = False

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    db_connection = get_db_connection()
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT id,temp,wind,rain,humidity,cloud,pressure,viscosity  FROM weather_data")
    weather_list = db_cursor.fetchall()
    df1 = pd.DataFrame(weather_list,
                       columns=["id", "temp", "wind", "rain", "humidity", "cloud", "pressure", "viscosity"])
    df1["viscosity"] = df1["viscosity"].apply(encode_viscosity)
    df1 = df1.fillna(df1.mean())
    df1 = df1.drop("id", axis=1)
    scaler = preprocessing.StandardScaler().fit(df1)
    data_scaled = scaler.transform(df1)
    df1 = pd.DataFrame(data=data_scaled, columns=df1.columns)
    print(df1)
    if find_k:

        distortions = []
        K = range(1, 50)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(df1)
            distortions.append(kmeanModel.inertia_)

        plt.figure(figsize=(16, 8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k for weather clusters')
        plt.show()

    else:

        kmeanModel = KMeans(n_clusters=5)
        kmeanModel.fit(df1)
        pickle.dump(kmeanModel, open("weather_cluster_model.sav", 'wb'))
        predicted = kmeanModel.predict(df1)
        print(predicted)
        df1['weather'] = predicted
        output_file = os.path.join(dirname, 'output\\weather_clusters.csv')

        # for row in df1.iterrows():
        #     print(row)
        # db_cursor.execute(f'UPDATE weather_data SET weather_category = {row[1][8]} '
        #                   f'WHERE id = {row[0]}')

        # db_connection.commit()
        df1.to_csv(output_file)

        # humidity, cloud, pressure
        x = "cloud"
        y = "pressure"

        plt.scatter(df1[x], df1[y], c=df1['weather'])
        plt.title("cluster visualization")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
