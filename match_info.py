def get_match_info(content, team):
    table_body = content.find_all("table", {"class": "match-details-table"})[0].find_all('tbody')

    match_info_table = table_body[0]
    rows = match_info_table.find_all('tr')
    match_info = {'Venue': content.find_all("td", {"class": "match-venue"})[0].get_text(), 'Toss_Won': 'N/A'}
    for row in rows:
        cols = row.find_all('td')
        if len(cols) == 2:
            match_info[cols[0].get_text()] = cols[1].get_text()
    if 'Toss' in match_info.keys():
        match_info['Toss_Won'] = team in match_info['Toss']

    return match_info
