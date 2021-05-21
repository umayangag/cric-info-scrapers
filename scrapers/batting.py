import pandas as pd
import re


def extract_batting_data(content, team):
    inning_index = -1

    headers = content.find_all("h5", {"class": "header-title label"})
    for i, header in enumerate(headers):
        if team + ' INNINGS' in header.get_text():
            inning_index = i

    batsmen_df = pd.DataFrame(
        columns=["Name", "Desc", "Runs", "Balls", "Minutes", "Fours", "Sixes", "Strike_Rate", "Batting_Position"])

    if inning_index > -1:
        table_body = content.find_all('tbody')

        for i, table in enumerate(table_body[0:4:2]):
            if i == inning_index:
                rows = table.find_all('tr')
                batting_position = 0
                for row in rows[::2]:
                    cols = row.find_all('td')
                    cols = [x.text.strip() for x in cols]
                    if cols[0] == 'Extras':
                        continue

                    if len(cols) > 7:
                        batting_position += 1
                        batsmen_df = batsmen_df.append(pd.Series(
                            [re.sub(r"\W+", ' ', cols[0].split("(c)")[0]).strip(), cols[1],
                             cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], batting_position],
                            index=batsmen_df.columns), ignore_index=True)

    return batsmen_df
