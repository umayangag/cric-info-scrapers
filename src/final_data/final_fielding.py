from config.mysql import get_db_connection
import pandas as pd
from final_data.queries import fielding_dataset_query
import os
from final_data.encoders import *
from team_selection.dataset_definitions import *

fielding_columns = input_fielding_columns

dirname = os.path.dirname(__file__)
output_file_encoded = os.path.join(dirname, "output\\fielding_encoded.csv")


# def calculate_fielding_performance(row):
#     return row['runs'] * row["strike_rate"]


# def categorize_fielding_performance(dataset):
#     fielding_performance = dataset[["runs", "strike_rate"]]
#     # calculate fielding performance
#     indexes = fielding_performance.apply(lambda row: calculate_fielding_performance(row),
#                                         axis=1)
#     fielding_performance["performance_index"] = indexes
#     fielding_performance = cluster_fielding_performance(fielding_performance)
#     print(fielding_performance)
#     dataset["performance"] = fielding_performance["fielding_performance"]
#     # dataset["performance_index"] = fielding_performance["performance_index"]
#     return dataset

def final_fielding_dataset(conn):
    db_cursor = conn.cursor()
    db_cursor.execute(fielding_dataset_query)
    data_list = db_cursor.fetchall()
    df_encoded = pd.DataFrame(data_list, columns=fielding_columns)
    # df_encoded = df_encoded.loc[df_encoded['runs_scored'] > 0]

    df_encoded["fielding_session"] = df_encoded["fielding_session"].apply(encode_session)
    df_encoded["fielding_viscosity"] = df_encoded["fielding_viscosity"].apply(encode_viscosity)
    df_encoded["success_rate"] = df_encoded.apply(lambda row:(row["catches"] + row["run_outs"]) / (
                    row["catches"] + row["run_outs"] + row["dropped_catches"] +
                    row["missed_run_outs"]), axis=1)
    # df_encoded["runs_scored"] = df_encoded["runs_scored"].apply(encode_runs)

    # df_encoded = categorize_fielding_performance(df_encoded)
    df_encoded = df_encoded.loc[:, df_encoded.columns != "player_name"]
    # df_encoded = normalize_fielding_dataset(df_encoded)
    # df_encoded = df_encoded.loc[:, df_encoded.columns != 'runs']
    # df_encoded = df_encoded.loc[:, df_encoded.columns != 'strike_rate']
    df_encoded = df_encoded.loc[:, df_encoded.columns != 'match_id']

    if os.path.exists(output_file_encoded):
        print("existing file deleted")
        os.remove(output_file_encoded)
    df_encoded.to_csv(output_file_encoded, index=False)
    print(fielding_dataset_query)


db_connection = get_db_connection()
final_fielding_dataset(db_connection)
