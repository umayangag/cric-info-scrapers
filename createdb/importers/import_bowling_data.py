import csv

from mysql.connector import utils
from ..config import get_db_connection
from datetime import datetime
from ..shared.utils import *

db_connection = get_db_connection()
db_cursor = db_connection.cursor()


def import_bowling_data(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'{", ".join(row)}')
            else:
                player = row[0]
                playerId = get_record_id("player", player)

                print(line_count)
                db_cursor.execute(f'INSERT INTO bowling_data SET'
                                  f' player_id={int(playerId)},'
                                  f' overs={float(row[1])},'
                                  f' balls={int(row[2])},'
                                  f' maidens={int(row[3])},'
                                  f' runs={int(row[4])},'
                                  f' wickets={int(row[5])},'
                                  f' econ={float(row[6])},'
                                  f' dots={int(row[7])},'
                                  f' fours={int(row[8])},'
                                  f' sixes={int(row[9])},'
                                  f' wides={int(row[10])},'
                                  f' no_balls={int(row[11])},'
                                  f' match_id={int(row[12])}'
                                  f'')
            line_count += 1
        db_connection.commit()
        print(f'Processed {line_count} lines.')
