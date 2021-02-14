def get_session_data(inning, hours):
    if 'Hours of play (local time)' not in hours.keys():
        return "", ""
    hours = hours['Hours of play (local time)']
    inning = int(inning)
    session_array = hours.split(',')
    if len(session_array) == 3:  # add missing comma to fix data
        session_array = hours.replace(" Interval", ", Interval", 1).split(',')
    batting_session = session_array[3].split(" ")[3]
    bowling_session = session_array[1].split(" ")[3]
    if inning == 1:
        batting_session = session_array[1].split(" ")[3]
        bowling_session = session_array[3].split(" ")[3]

    return batting_session, bowling_session
