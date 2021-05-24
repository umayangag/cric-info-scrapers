import csv

from shared.utils import *


def import_batting_data(db_connection, filename):
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
                db_cursor.execute(f'INSERT INTO batting_data SET'
                                  f' player_id={int(playerId)},'
                                  f' description="{row[1]}",'
                                  f' runs={int(row[2])},'
                                  f' balls={int(row[3])},'
                                  f' minutes={int_or_default(row[4])},'
                                  f' fours={int(row[5])},'
                                  f' sixes={int(row[6])},'
                                  f' strike_rate={float(row[7])},'
                                  f' batting_position={int(row[8])},'
                                  f' match_id={int(row[9])}'
                                  f'')
            line_count += 1
        db_connection.commit()
        print(f'Processed {line_count} lines.')
