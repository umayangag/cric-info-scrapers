import csv
import pandas as pd
import matplotlib.pyplot as plt

batting_data_df = pd.read_csv("data/batting_data.csv")
bowling_data_df = pd.read_csv("data/bowling_data.csv")

# plt.matshow(batting_data_df.corr())
# plt.show()

# print(batting_data_df.corr())
# batting_data_df.corr().to_csv("results/correlations.csv")

plt.hist(batting_data_df['strike_rate'], color='blue', edgecolor='black',
         bins=int(180 / 5))
plt.show()
