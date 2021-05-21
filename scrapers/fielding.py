import pandas as pd
import re


def extract_fielding_data(content, team):
    inning_index = -1

    headers = content.find_all("h5", {"class": "header-title label"})
    for i, header in enumerate(headers):
        if ' INNINGS' in header.get_text() and team + ' INNINGS' not in header.get_text():
            inning_index = i

    fielding_dict = {}
    fielding_df = pd.DataFrame(
        columns=["Name", "Catches", "Run Outs"])

    if inning_index > -1:
        table_body = content.find_all('tbody')

        for i, table in enumerate(table_body[0:4:2]):
            if i == inning_index:
                rows = table.find_all('tr')
                for row in rows[::2]:
                    cols = row.find_all('td')
                    cols = [x.text.strip() for x in cols]
                    if cols[0] == 'Extras':
                        continue

                    if len(cols) > 7:
                        fielding_data = cols[1]
                        if fielding_data.startswith("c & b "):
                            value = fielding_data.split(" b ")[1].strip()
                            if value in fielding_dict.keys():
                                fielding_dict[value]["catches"] += 1
                            else:
                                fielding_dict[value] = {"catches": 1, "run outs": 0}

                        elif fielding_data.startswith("c "):
                            value = fielding_data.split(" b ")[0].replace("c ", "", 1).strip()
                            if value in fielding_dict.keys():
                                fielding_dict[value]["catches"] += 1
                            else:
                                fielding_dict[value] = {"catches": 1, "run outs": 0}

                        elif fielding_data.startswith("st "):
                            value = fielding_data.split(" b ")[0].replace("st ", "", 1).strip()
                            if value in fielding_dict.keys():
                                fielding_dict[value]["run outs"] += 1
                            else:
                                fielding_dict[value] = {"catches": 0, "run outs": 1}

                        elif fielding_data.startswith("run out") or fielding_data.startswith("st "):
                            values = fielding_data.replace("run out", "", 1).replace("(", "", 1).replace(")", "",
                                                                                                         1).split("/")

                            for value in values:
                                value = value.strip()
                                if value in fielding_dict.keys():
                                    fielding_dict[value]["run outs"] += 1
                                else:
                                    fielding_dict[value] = {"catches": 0, "run outs": 1}
    for key in fielding_dict.keys():
        fielding_df = fielding_df.append(pd.Series(
            [key, fielding_dict[key]["catches"], fielding_dict[key]["run outs"]],
            index=fielding_df.columns), ignore_index=True)
    return fielding_df
