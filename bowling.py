import pandas as pd


def extract_bowling_data(content, team):
    inning_index = -1

    headers = content.find_all("h5", {"class": "header-title label"})
    for i, header in enumerate(headers):
        if 'INNINGS' in header.get_text() and team not in header.get_text():
            inning_index = i
    bowler_df = pd.DataFrame(columns=['Name', 'Overs', 'Maidens', 'Runs', 'Wickets',
                                      'Econ', 'Dots', '4s', '6s', 'Wd', 'Nb'])
    if inning_index > -1:
        table_body = content.find_all('tbody')

        for i, table in enumerate(table_body[1:4:2]):
            if i == inning_index:
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    cols = [x.text.strip() for x in cols]
                    bowler_df = bowler_df.append(pd.Series([cols[0], cols[1], cols[2], cols[3], cols[4], cols[5],
                                                            cols[6], cols[7], cols[8], cols[9], cols[10]],
                                                           index=bowler_df.columns), ignore_index=True)
    return bowler_df
