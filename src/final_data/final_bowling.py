from config.mysql import get_db_connection
import pandas as pd
from final_data.queries import bowling_dataset_query
import os
from analyze.cluster_bowling import cluster_bowling_performance
from analyze.normalize_bowling import normalize_bowling_dataset
from sklearn.cluster import KMeans
from final_data.encoders import *

columns = [
    "econ",
    "wickets",
    "match_id",
    "player_name",
    "player_consistency",
    "player_form",
    "temp",
    "wind",
    "rain",
    "humidity",
    "cloud",
    "pressure",
    "viscosity",
    "inning",
    "bowling_session",
    "toss",
    "venue",
    "opposition",
    "season",
]

dirname = os.path.dirname(__file__)
output_file_encoded = os.path.join(dirname, "output\\bowling_encoded.csv")


def categorize_bowling_performance(dataset):
    bowling_performance = dataset[["econ", "wickets"]]
    # calculate bowling performance
    bowling_performance = cluster_bowling_performance(bowling_performance)
    print(bowling_performance)
    dataset["performance"] = bowling_performance["bowling_performance"]
    # dataset["performance_index"] = bowling_performance["performance_index"]
    return dataset


def final_bowling_dataset(conn):
    db_cursor = conn.cursor()
    db_cursor.execute(bowling_dataset_query)
    data_list = db_cursor.fetchall()
    df_encoded = pd.DataFrame(data_list, columns=columns)

    df_encoded["bowling_session"] = df_encoded["bowling_session"].apply(encode_session)
    df_encoded["viscosity"] = df_encoded["viscosity"].apply(encode_viscosity)
    # df_encoded["econ"] = df_encoded["econ"].apply(encode_econ)

    # df_encoded = categorize_bowling_performance(df_encoded)
    df_encoded = df_encoded.loc[:, df_encoded.columns != "player_name"]
    # df_encoded = normalize_bowling_dataset(df_encoded)
    # df_encoded = df_encoded.loc[:, df_encoded.columns != 'econ']
    # df_encoded = df_encoded.loc[:, df_encoded.columns != 'wickets']
    df_encoded = df_encoded.loc[:, df_encoded.columns != 'match_id']

    if os.path.exists(output_file_encoded):
        print("existing file deleted")
        os.remove(output_file_encoded)
    df_encoded.to_csv(output_file_encoded, index=False)
    print(bowling_dataset_query)


db_connection = get_db_connection()
final_bowling_dataset(db_connection)
