from newspaper import Article
import sqlite3
import requests
from bs4 import BeautifulSoup
from datetime import datetime

class NewsCrawler:
    def __init__(self, db_name='news_articles.db'):
        self.db_name = db_name
        self.setup_database()
        
    def setup_database(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS articles
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      title TEXT,
                      text TEXT,
                      url TEXT UNIQUE,
                      source TEXT,
                      publish_date DATETIME,
                      retrieved_date DATETIME)''')
        conn.commit()
        conn.close()
        
    def crawl_article(self, url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            now = datetime.now()
            c.execute('''INSERT OR IGNORE INTO articles 
                        (title, text, url, source, publish_date, retrieved_date)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                     (article.title,
                      article.text,
                      url,
                      article.source_url,
                      article.publish_date,
                      now))
            
            conn.commit()
            conn.close()
            print(f"Successfully crawled: {article.title}")
            
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
            
    def crawl_news_site(self, base_url, article_link_selector):
        try:
            response = requests.get(base_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all article links using the provided CSS selector
            article_links = soup.select(article_link_selector)
            
            for link in article_links:
                if 'href' in link.attrs:
                    article_url = link['href']
                    if not article_url.startswith('http'):
                        article_url = base_url + article_url
                    self.crawl_article(article_url)
                    
        except Exception as e:
            print(f"Error crawling {base_url}: {str(e)}")

# Example usage
if __name__ == "__main__":
    crawler = NewsCrawler()
    
    # Example for Reuters
    crawler.crawl_news_site(
        "https://www.reuters.com",
        "a.story-card__headline__link",  # CSS selector for article links
    )