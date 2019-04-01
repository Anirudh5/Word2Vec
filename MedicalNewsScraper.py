from urllib import request
from bs4 import BeautifulSoup

def get_page_content(url):
    page = request.urlopen(url).read().decode("utf-8", "ignore")
    soup = BeautifulSoup(page, "html.parser")
    article_body = soup.find("div", itemprop="articleBody")
    try:
        content = article_body.find("header").text
    except:
        print(url)
        content = ""
    paragraphs = article_body.find_all("p")
    for p in paragraphs: content += p.text
    content = content.strip()
    outfile.write(content)

def get_article_links(page_url):
    page = request.urlopen(page_url).read().decode("utf-8", "ignore")
    soup = BeautifulSoup(page, "html.parser")
    listing = soup.find("ul", class_="listing")
    articles = listing.find_all("a")
    for article in articles:
        url = "https://www.medicalnewstoday.com" + article.get('href')
        get_page_content(url)

if __name__ == '__main__':
    outfile = open("medical_news.txt", "w")
    base_url = "https://www.medicalnewstoday.com/archive/"
    for i in range(45, -1, -1):
        if i == 0: page_url = base_url
        else: page_url = base_url + str(i)
        get_article_links(page_url)
        print("Page", i, "done")
