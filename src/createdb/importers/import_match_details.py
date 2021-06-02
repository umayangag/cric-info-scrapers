import csv

from datetime import datetime
from shared.utils import *


def import_match_details(db_connection, filename):
    db_cursor = db_connection.cursor()
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'{", ".join(row)}')
            else:
                opposition = row[8]
                oppositionId = get_record_id(db_connection, "opposition", opposition)

                venue = row[14]
                venueId = get_record_id(db_connection, "venue", venue)

                date_value = datetime.strptime(row[9], '%d-%b-%y')

                season = row[18]
                seasonId = get_record_id(db_connection, "season", season)

                toss = 0
                if row[17] == "TRUE":
                    toss = 1

                result = -1
                if row[7] == "won":
                    result = 1
                elif row[7] == "lost":
                    result = 0

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
                                  f' inning={int_or_default(row[6])},'
                                  f' result={int_or_default(result)},'
                                  f' opposition_id={oppositionId},'
                                  f' date="{date_value}",'
                                  f' match_id={int(row[10])},'
                                  f' batting_session="{batting_session_label}",'
                                  f' bowling_session="{bowling_session_label}",'
                                  f' venue_id={venueId},'
                                  f' extras={int_or_default(row[16])},'
                                  f' toss="{toss}",'
                                  f' season_id={seasonId}, '
                                  f' match_number={int(match_number)}'
                                  f'')
            line_count += 1
        db_connection.commit()
        print(f'Processed {line_count} lines.')
