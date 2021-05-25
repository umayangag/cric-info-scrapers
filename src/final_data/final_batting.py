from config.mysql import get_db_connection
import pandas as pd
from final_data.queries import batting_dataset_query
import os

columns = [
    "runs",
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
]

dirname = os.path.dirname(__file__)
output_file = os.path.join(dirname, "output\\batting.csv")
output_file_encoded = os.path.join(dirname, "output\\batting_encoded.csv")


def encode_session(value):
    if value == "day":
        return 0
    return 1


def encode_viscosity(value):
    if value == "Excellent":
        return 3
    if value == "Good":
        return 2
    if value == "Average":
        return 1
    return 0


def final_batting_dataset(conn):
    db_cursor = conn.cursor()
    db_cursor.execute(batting_dataset_query)
    data_list = db_cursor.fetchall()
    df = pd.DataFrame(data_list, columns=columns)

    df_encoded = df.copy()
    df_encoded["batting_session"] = df_encoded["batting_session"].apply(encode_session)
    df_encoded["viscosity"] = df_encoded["viscosity"].apply(encode_viscosity)
    df_encoded = df_encoded.loc[:, df_encoded.columns != "player_name"]

    df_encoded.to_csv(output_file, index=False)
    df_encoded.to_csv(output_file_encoded, index=False)
    print(batting_dataset_query)


db_connection = get_db_connection()
final_batting_dataset(db_connection)
