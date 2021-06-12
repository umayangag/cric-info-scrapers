from config.mysql import get_db_connection
from preprocessing.calculate_bowling_form import calculate_bowling_form
from preprocessing.calculate_bowling_venue import calculate_bowling_venue
from preprocessing.calculate_bowling_opposition import calculate_bowling_opposition
from preprocessing.calculate_bowling_consistency import calculate_bowling_consistency

from preprocessing.calculate_batting_form import calculate_batting_form
from preprocessing.calculate_batting_venue import calculate_batting_venue
from preprocessing.calculate_batting_opposition import calculate_batting_opposition
from preprocessing.calculate_batting_consistency import calculate_batting_consistency
from preprocessing.calculate_fielding_consistency import calculate_fielding_consistency


db_connection = get_db_connection()

calculate_bowling_opposition(db_connection)
calculate_bowling_venue(db_connection)
calculate_bowling_consistency(db_connection)
calculate_bowling_form(db_connection)

calculate_batting_venue(db_connection)
calculate_batting_opposition(db_connection)
calculate_batting_form(db_connection)
calculate_batting_consistency(db_connection)
calculate_fielding_consistency(db_connection)
