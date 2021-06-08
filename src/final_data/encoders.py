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


hash_runs = {0: 30, 1: 60, 2: 90, 3: 200}
hash_balls_faced = {0: 30, 1: 60, 2: 90, 3: 200}
hash_batting_position = {0: 4, 1: 7, 2: 12}
hash_fours = {0: 4, 1: 8, 2: 12}
hash_sixes = {0: 2, 1: 4, 2: 6}


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
    if value < 25:
        return 0
    if value < 43:
        return 1
    if value < 61:
        return 2
    return 3


def encode_deliveries_bowled(value):
    if value < 22:
        return 0
    if value < 38:
        return 1
    if value < 52:
        return 2
    return 3


def encode_econ(value):
    if value < 4:
        return 0
    if value < 8:
        return 1
    if value < 12:
        return 2
    return 3


def encode_wickets(value):
    if value < 1:
        return 0
    if value < 2:
        return 1
    if value < 3:
        return 2
    if value < 4:
        return 3
    return 4
