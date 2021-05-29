from config.mysql import get_db_connection
import pandas as pd
from final_data.queries import batting_win_dataset_query
import os
from analyze.cluster_batting import cluster_batting_performance
from analyze.normalize_dataset import normalize_dataset
from final_data.encoders import *
from sklearn import preprocessing

columns = [
    "total_score",
    "total_balls",
    "runs",
    "balls",
    "fours",
    "sixes",
    "strike_rate",
    "match_id",
    "batting_position",
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
    "batting_session",
    "toss",
    "venue",
    "opposition",
    "season",
    "result",
]

dirname = os.path.dirname(__file__)
output_file_encoded = os.path.join(dirname, "output\\batting_win_predict_encoded.csv")


def calculate_contribution(row):
    return row["runs"] / row["total_score"]


def final_batting_dataset(conn):
    db_cursor = conn.cursor()
    db_cursor.execute(batting_win_dataset_query)
    data_list = db_cursor.fetchall()
    df_encoded = pd.DataFrame(data_list, columns=columns)

    df_encoded["batting_session"] = df_encoded["batting_session"].apply(encode_session)
    df_encoded["viscosity"] = df_encoded["viscosity"].apply(encode_viscosity)
    # df_encoded["performance"] = df_encoded["runs"].apply(encode_runs)
    df_encoded["contribution"] = df_encoded.apply(lambda row: calculate_contribution(row), axis=1)

    df_encoded = df_encoded.loc[:, df_encoded.columns != "player_name"]
    # df_encoded = normalize_dataset(df_encoded)
    scaler = preprocessing.StandardScaler().fit(df_encoded)
    data_scaled = scaler.transform(df_encoded)
    final_df = pd.DataFrame(data=data_scaled, columns=df_encoded.columns)
    final_df["result"] = df_encoded["result"]

    if os.path.exists(output_file_encoded):
        print("existing file deleted")
        os.remove(output_file_encoded)
    final_df.to_csv(output_file_encoded, index=False)
    print(batting_win_dataset_query)


db_connection = get_db_connection()
final_batting_dataset(db_connection)
