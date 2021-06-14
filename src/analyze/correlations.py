import pandas as pd

batting_data_df = pd.read_csv("data/batting_data.csv")
bowling_data_df = pd.read_csv("data/bowling_data.csv")


# print(batting_data_df.corr())
batting_data_df.corr().to_csv("results/batting-correlations.csv")
bowling_data_df.corr().to_csv("results/bowling-correlations.csv")
