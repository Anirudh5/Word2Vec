import re
import requests
import threading
from bs4 import BeautifulSoup

class Scraper(threading.Thread):
    def __init__(self, url, file):
        threading.Thread.__init__(self)
        self.page_url = url
        self.file = open(str(i) + ".txt", "w")

    def run(self):
        page = requests.get(self.page_url, {'User-Agent': 'Mozilla/5.0'}).text
        soup = BeautifulSoup(page, "html.parser")
        listing = soup.find("ul", class_="listing")
        articles = listing.find_all("a")
        for article in articles:
            url = "https://www.medicalnewstoday.com" + article.get('href')
            try: self.get_page_content(url)
            except Exception as e: print(url)

    def get_page_content(self, url):
        page = requests.get(url, {'User-Agent': 'Mozilla/5.0'}).text
        soup = BeautifulSoup(page, "html.parser")
        article_body = soup.find("div", itemprop="articleBody")
        content = article_body.text
        content = re.sub("Table of contents", "", content)
        content = " ".join(content.split())
        content = re.sub("googleAdSlotInfo(.*);", "", content)
        self.file.write(content + "\n\n")

if __name__ == '__main__':
    base_url = "https://www.medicalnewstoday.com/archive/"
    for i in range(45, -1, -1):
        if i == 0: page_url = base_url
        else: page_url = base_url + str(i)
        Scraper(page_url, i).start()
