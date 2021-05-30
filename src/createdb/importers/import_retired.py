import pandas as pd

# UPDATE `player` SET `is_retired` = '0'
# UPDATE `player` SET `is_wicket_keeper` = '0'

def import_retired_data(db_connection, filename):
    retired = pd.read_csv(filename)
    db_cursor = db_connection.cursor()
    for index, row in retired.iterrows():
        name = row['Name']
        db_cursor.execute(f'UPDATE player SET is_retired = 1 WHERE player.player_name LIKE "{name}"')
    db_connection.commit()
