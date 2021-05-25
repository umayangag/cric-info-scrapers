from config.mysql import get_db_connection
import pandas as pd
from final_data.queries import batting_dataset_query
import os

columns = [
    "match_id",
    "runs",
    "strike_rate",
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


def final_batting_dataset(conn):
    db_cursor = conn.cursor()
    db_cursor.execute(batting_dataset_query)
    data_list = db_cursor.fetchall()
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv(output_file, index=False)
    print(batting_dataset_query)


db_connection = get_db_connection()
final_batting_dataset(db_connection)
