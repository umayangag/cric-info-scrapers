def encode_session(value):
    if value == "day":
        return 1
    return 2


def encode_viscosity(value):
    if value == "Excellent":
        return 4
    if value == "Good":
        return 3
    if value == "Average":
        return 2
    return 1


def encode_how_out(value):
    if value == 0:
        return 0
    if value == "not out":
        return 1
    return 2


def encode_runs(value):
    if value < 25:
        return 1
    if value < 50:
        return 2
    if value < 75:
        return 3
    return 4


def encode_econ(value):
    if value < 4:
        return 1
    if value < 8:
        return 2
    if value < 12:
        return 3
    return 4
