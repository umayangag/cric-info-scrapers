import csv

from shared.utils import *


def import_fielding_data(db_connection, filename):
    db_cursor = db_connection.cursor()
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'{", ".join(row)}')
            else:
                player = row[0]
                playerId = get_record_id(db_connection, "player", player)

                print(line_count)
                db_cursor.execute(f'INSERT INTO fielding_data SET'
                                  f' player_id={int(playerId)},'
                                  f' catches={int(row[1])},'
                                  f' run_outs={int(row[2])},'
                                  f' dropped_catches={int(row[3])},'
                                  f' missed_run_outs={int(row[4])},'
                                  f' match_id={int(row[5])}'
                                  f'')
            line_count += 1
        db_connection.commit()
        print(f'Processed {line_count} lines.')
