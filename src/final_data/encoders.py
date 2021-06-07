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


def encode_fours(value):
    if value < 1:
        return 0
    if value < 2:
        return 1
    if value < 4:
        return 2
    if value < 8:
        return 3
    return 4


def encode_fours(value):
    if value < 2:
        return 0
    if value < 4:
        return 1
    if value < 7:
        return 2
    if value < 11:
        return 3
    return 4


def encode_runs(value):
    if value < 14:
        return 0
    if value < 33:
        return 1
    if value < 59:
        return 2
    if value < 92:
        return 3
    return 4


def encode_balls_faced(value):
    if value < 17:
        return 0
    if value < 39:
        return 1
    if value < 65:
        return 2
    if value < 97:
        return 3
    return 4


def encode_runs_conceded(value):
    if value < 25:
        return 0
    if value < 43:
        return 1
    if value < 61:
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
