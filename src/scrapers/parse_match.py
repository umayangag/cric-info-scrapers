import requests
from bs4 import BeautifulSoup


def get_content(match_id):
    URL = 'https://stats.espncricinfo.com/ci/engine/match/' + str(match_id) + '.html'
    page = requests.get(URL)
    return BeautifulSoup(page.content, 'html.parser')
