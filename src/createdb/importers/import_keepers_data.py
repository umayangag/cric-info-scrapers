import pandas as pd


def import_keepers_data(db_connection, filename):
    keepers = pd.read_csv(filename)
    db_cursor = db_connection.cursor()
    for index, row in keepers.iterrows():
        name = row['Name']
        db_cursor.execute(f'UPDATE player SET is_wicket_keeper = 1 WHERE player.player_name LIKE "{name}"')
    db_connection.commit()
