import parse_match, match_list, match_info, batting, bowling, session, fielding
import pandas as pd

team = "Sri Lanka"
match_list = match_list.extract_match_list(
    'https://stats.espncricinfo.com/ci/engine/team/8.html?class=2;spanmax1=01+Jan+2020;spanmin1=01+Jan+2010;spanval1=span;template=results;type=team;view=innings')

batting_data = pd.DataFrame()
bowling_data = pd.DataFrame()
fielding_data = pd.DataFrame()

for index, match in match_list.iterrows():
    print(match_list.at[index, 'Match_Id'])
    match_content = parse_match.get_content(match_list.at[index, 'Match_Id'])
    match_data = match_info.get_match_info(match_content, team)
    batting_session, bowling_session = session.get_session_data(match_list.at[index, 'Inning'], match_data)
    match_list.at[index, "Batting_Session"] = batting_session
    match_list.at[index, "Bowling_Session"] = bowling_session
    for key in match_data.keys():
        match_list.at[index, key] = match_data[key]

    # batting_stats = batting.extract_batting_data(match_content, team)
    # batting_stats['Match_Id'] = match_list.at[index, 'Match_Id']
    # batting_data = batting_data.append(batting_stats)

    # bowling_stats = bowling.extract_bowling_data(match_content, team)
    # bowling_stats['Match_Id'] = match_list.at[index, 'Match_Id']
    # bowling_data = bowling_data.append(bowling_stats)

    fielding_stats = fielding.extract_fielding_data(match_content, team)
    fielding_stats['Match_Id'] = match_list.at[index, 'Match_Id']
    fielding_data = fielding_data.append(fielding_stats)

# match_list.to_csv(path_or_buf='extracted/match_info.csv', index=False, sep=';')
# batting_data.to_csv(path_or_buf='extracted/batting.csv', index=False, sep=';')
fielding_data.to_csv(path_or_buf='extracted/fielding.csv', index=False, sep=';')
# bowling_data.to_csv(path_or_buf='extracted/bowling.csv', index=False, sep=';')
