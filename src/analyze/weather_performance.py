import math
import os

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    # filename = os.path.join(dirname, 'cluster_data\\batting_performance.csv')
    filename = os.path.join(dirname, 'data\\batting_encoded.csv')

    batting_data= pd.read_csv(filename)

    for attr in ['batting_position', 'batting_consistency', 'batting_form',
       'batting_temp', 'batting_wind', 'batting_rain', 'batting_humidity',
       'batting_cloud', 'batting_pressure', 'batting_viscosity',
       'batting_inning', 'batting_session', 'toss', 'venue', 'opposition',
       'season']:
        if attr != "runs_scored":
            df=batting_data.copy()
            min = df[attr].min()
            max = df[attr].max()
            interval =2
            print(attr, min, max, interval)
            df['bin'] = pd.cut(df[attr], bins=range(math.floor(min), math.ceil(max + 1), interval))
            df = df.groupby('bin').mean()

            plt.plot(df[attr], df["runs_scored"])
            plt.title("Batting Average vs " + attr)
            plt.xlabel(attr)
            plt.ylabel("Batting Average")
            plt.show()
