from config.mysql import get_db_connection
import pandas as pd
from final_data.queries import batting_dataset_query
import os
from final_data.encoders import *
from team_selection.dataset_definitions import *

batting_columns = np.concatenate((output_batting_columns, input_batting_columns))

dirname = os.path.dirname(__file__)
output_file_encoded = os.path.join(dirname, "output\\batting_encoded.csv")


def fill_batting_form(row):
    if row["batting_form"] == 0:
        return row["batting_consistency"]
    return row["batting_form"]


# def calculate_batting_performance(row):
#     return row['runs'] * row["strike_rate"]


# def categorize_batting_performance(dataset):
#     batting_performance = dataset[["runs", "strike_rate"]]
#     # calculate batting performance
#     indexes = batting_performance.apply(lambda row: calculate_batting_performance(row),
#                                         axis=1)
#     batting_performance["performance_index"] = indexes
#     batting_performance = cluster_batting_performance(batting_performance)
#     print(batting_performance)
#     dataset["performance"] = batting_performance["batting_performance"]
#     # dataset["performance_index"] = batting_performance["performance_index"]
#     return dataset

def final_batting_dataset(conn):
    db_cursor = conn.cursor()
    db_cursor.execute(batting_dataset_query)
    data_list = db_cursor.fetchall()
    df_encoded = pd.DataFrame(data_list, columns=batting_columns)
    # df_encoded = df_encoded.loc[df_encoded['runs_scored'] > 0]
    df_encoded["batting_form"] = df_encoded.apply(lambda x: fill_batting_form(x), axis=1)
    df_encoded["batting_session"] = df_encoded["batting_session"].apply(encode_session)
    df_encoded["batting_viscosity"] = df_encoded["batting_viscosity"].apply(encode_viscosity)
    # df_encoded["runs_scored"] = df_encoded["runs_scored"].apply(encode_runs)

    # df_encoded = categorize_batting_performance(df_encoded)
    df_encoded = df_encoded.loc[:, df_encoded.columns != "player_name"]
    # df_encoded = normalize_batting_dataset(df_encoded)
    # df_encoded = df_encoded.loc[:, df_encoded.columns != 'runs']
    # df_encoded = df_encoded.loc[:, df_encoded.columns != 'strike_rate']
    df_encoded = df_encoded.loc[:, df_encoded.columns != 'match_id']

    if os.path.exists(output_file_encoded):
        print("existing file deleted")
        os.remove(output_file_encoded)
    df_encoded.to_csv(output_file_encoded, index=False)
    print(batting_dataset_query)


db_connection = get_db_connection()
final_batting_dataset(db_connection)
