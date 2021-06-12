from config.mysql import get_db_connection
import pandas as pd
from final_data.queries import bowling_dataset_query
import os
from analyze.cluster_bowling import cluster_bowling_performance
from analyze.normalize_bowling import normalize_bowling_dataset
from sklearn.cluster import KMeans
from final_data.encoders import *
from team_selection.dataset_definitions import *

bowling_columns = np.concatenate((output_bowling_columns, input_bowling_columns))

dirname = os.path.dirname(__file__)
output_file_encoded = os.path.join(dirname, "output\\bowling_encoded.csv")


# def categorize_bowling_performance(dataset):
#     bowling_performance = dataset[["econ", "wickets"]]
#     # calculate bowling performance
#     bowling_performance = cluster_bowling_performance(bowling_performance)
#     print(bowling_performance)
#     dataset["performance"] = bowling_performance["bowling_performance"]
#     # dataset["performance_index"] = bowling_performance["performance_index"]
#     return dataset
def fill_bowling_form(row):
    if row["bowling_form"] == 0:
        return row["bowling_consistency"]
    return row["bowling_form"]


def final_bowling_dataset(conn):
    db_cursor = conn.cursor()
    db_cursor.execute(bowling_dataset_query)
    data_list = db_cursor.fetchall()
    df_encoded = pd.DataFrame(data_list, columns=bowling_columns)

    df_encoded["bowling_form"] = df_encoded.apply(lambda x: fill_bowling_form(x), axis=1)
    df_encoded["bowling_session"] = df_encoded["bowling_session"].apply(encode_session)
    df_encoded["bowling_viscosity"] = df_encoded["bowling_viscosity"].apply(encode_viscosity)
    # df_encoded["runs_conceded"] = df_encoded["runs_conceded"].apply(encode_runs_conceded)

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
