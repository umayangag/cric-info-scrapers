import csv

from mysql.connector import utils
from ..config import get_db_connection
from datetime import datetime
from ..shared.utils import *

db_connection = get_db_connection()
db_cursor = db_connection.cursor()


def import_weather_data(filename, session):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'{", ".join(row)}')
            else:

                temp = row[5].replace("°c", "", 1)
                feels = row[6].replace("°c", "", 1)
                wind = row[7].split(" ", 1)[0]
                gust = row[8].split(" ", 1)[0]
                rain = row[9].split(" ", 1)[0]
                humidity = row[10].replace("%", "", 1)
                cloud = row[11].replace("%", "", 1)
                pressure = row[12].split(" ", 1)[0]
                viscosity = row[13].replace("°c", "", 1)

                print(line_count)
                db_cursor.execute(f'INSERT INTO weather_data SET'
                                  f' match_id={int(row[1])},'
                                  f' session="{session}",'
                                  f' temp={int_or_default(temp)},'
                                  f' feels={int_or_default(feels)},'
                                  f' wind={int_or_default(wind)},'
                                  f' gust={int_or_default(gust)},'
                                  f' rain={float_or_default(rain)},'
                                  f' humidity={int_or_default(humidity)},'
                                  f' cloud={int_or_default(cloud)},'
                                  f' pressure={int_or_default(pressure)},'
                                  f' viscosity="{viscosity}"'
                                  f'')
            line_count += 1
        db_connection.commit()
        print(f'Processed {line_count} lines.')
