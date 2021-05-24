import csv

from mysql.connector import utils
from ..config import get_db_connection
from datetime import datetime
from ..shared.utils import *

db_connection = get_db_connection()
db_cursor = db_connection.cursor()


def import_match_details(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'{", ".join(row)}')
            else:
                opposition = row[8]
                oppositionId = get_record_id("opposition", opposition)

                venue = row[14]
                venueId = get_record_id("venue", venue)

                date_value = datetime.strptime(row[9], '%d-%b-%y')

                season = row[18]
                seasonId = get_record_id("season", season)

                toss = "lost"
                if row[17] == "TRUE":
                    toss = "won"

                match_number = row[19].replace("ODI No. ", "", 1)

                batting_session_time = int_or_default(
                    row[12].replace(":", ".", 1).split(".", 1)[0])
                if batting_session_time == "NULL":
                    batting_session_label = ""
                elif batting_session_time >= 18:
                    batting_session_label = "night"
                else:
                    batting_session_label = "day"

                bowling_session_time = int_or_default(
                    row[13].replace(":", ".", 1).split(".", 1)[0])
                if bowling_session_time == "NULL":
                    bowling_session_label = ""
                elif bowling_session_time >= 18:
                    bowling_session_label = "night"
                else:
                    bowling_session_label = "day"

                print(toss)

                db_cursor.execute(f'INSERT INTO match_details SET'
                                  f' score={int_or_default(row[0])},'
                                  f' wickets={int_or_default(row[1])},'
                                  f' overs={float_or_default(row[2])},'
                                  f' balls={int_or_default(row[3])},'
                                  f' rpo={float_or_default(row[4])},'
                                  f' target={int_or_default(row[5])},'
                                  f' inning="{"inning"+row[6]}",'
                                  f' result="{row[7]}",'
                                  f' opposition_id={oppositionId},'
                                  f' date="{date_value}",'
                                  f' match_id={int(row[10])},'
                                  f' batting_session="{batting_session_label}",'
                                  f' bowling_session="{bowling_session_label}",'
                                  f' venue_id="{venueId}",'
                                  f' extras={int_or_default(row[16])},'
                                  f' toss="{toss}",'
                                  f' season_id={seasonId}, '
                                  f' match_number={int(match_number)}'
                                  f'')
            line_count += 1
        db_connection.commit()
        print(f'Processed {line_count} lines.')
