def encode_session(value):
    if value == "day":
        return 0
    return 1


def encode_viscosity(value):
    if value == "Excellent":
        return 3
    if value == "Good":
        return 2
    if value == "Average":
        return 1
    return 0


def encode_how_out(value):
    if value == 0:
        return 0
    if value == "not out":
        return 1
    return 2


hash_runs = {0: 30, 1: 60, 2: 90, 3: 120}
hash_runs_conceded = {0: 40, 1: 60, 2: 80}
hash_balls_faced = {0: 30, 1: 60, 2: 90, 3: 120}
hash_batting_position = {0: 4, 1: 7, 2: 12}
hash_fours = {0: 4, 1: 8, 2: 12}
hash_sixes = {0: 2, 1: 4, 2: 6}
hash_wickets_taken = {0: 2, 1: 5, 2: 10}
hash_economy = {0: 4, 1: 8, 2: 20}
hash_deliveries = {0: 30, 1: 61}


def decode_deliveries(value):
    return decode_value(hash_deliveries, value)


def decode_runs_conceded(value):
    return decode_value(hash_runs_conceded, value)


def decode_wickets_taken(value):
    return decode_value(hash_wickets_taken, value)


def decode_balls_faced(value):
    return decode_value(hash_balls_faced, value)


def decode_runs(value):
    return decode_value(hash_runs, value)


def decode_value(value_array, value):
    count = len(value_array) - 1
    # if value > count:
    #     return value_array[count]
    if value == 0:
        return value_array[0] / 2
    return value_array[value] / 2


def encode_value(value_array, value):
    count = len(value_array)
    for i in range(count):
        if value < value_array[i]:
            return i
    return count - 1


def encode_batting_position(value):
    return encode_value(hash_batting_position, value)


def encode_runs(value):
    return encode_value(hash_runs, value)


def encode_balls_faced(value):
    return encode_value(hash_balls_faced, value)


def encode_sixes(value):
    return encode_value(hash_sixes, value)


def encode_fours(value):
    return encode_value(hash_fours, value)


def encode_runs_conceded(value):
    return encode_value(hash_runs_conceded, value)


def encode_deliveries_bowled(value):
    return encode_value(hash_deliveries, value)


def encode_econ(value):
    return encode_value(hash_economy, value)


def encode_wickets(value):
    return encode_value(hash_wickets_taken, value)
