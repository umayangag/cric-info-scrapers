import requests
from bs4 import BeautifulSoup
import pandas as pd


def extract_match_list(url):
    page = requests.get(url)
    bs = BeautifulSoup(page.content, 'html.parser')

    table_body = bs.find_all('tbody')
    match_list = pd.DataFrame(
        columns=["Score", "Wickets", "Overs", "RPO", "Target", "Inning", "Result", "Opposition", "Ground", "Date",
                 "Match_Id", "URL_Text"])

    rows = table_body[2].find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        score_array = cols[0].text.strip().split('/')
        wickets = 10
        if score_array[0] == "DNB":
            wickets = 0
        if len(score_array) == 2:
            wickets = score_array[1]

        match_list = match_list.append(pd.Series(
            [score_array[0],
             wickets,
             cols[1].text.strip(),
             cols[2].text.strip(),
             cols[3].text.strip(),
             cols[4].text.strip(),
             cols[5].text.strip(),
             cols[7].find_all('a')[0].get_text(),
             cols[8].text.strip(),
             cols[9].text.strip(),
             cols[10].find_all('a')[0].get('href').split('/')[4].replace('.html', ''),
             cols[10].find_all('a')[0].get_text()],
            index=match_list.columns), ignore_index=True)

    return match_list
