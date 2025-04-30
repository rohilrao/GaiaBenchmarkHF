import scrapy
from scrapy.crawler import CrawlerProcess
from duckduckgo_search import DDGS as ddg
from scrapy.http import Request
from scrapy.utils.project import get_project_settings
from scrapy import signals
from collections import defaultdict
import logging
from pydispatch import dispatcher
from tqdm import tqdm
import re
import html

class SearchSpider(scrapy.Spider):
    name = 'search_spider'
    
    def __init__(self, search_results=None, *args, **kwargs):
        super(SearchSpider, self).__init__(*args, **kwargs)
        self.search_results = search_results or []
        self.parsed_content = defaultdict(str)
        self.logger.setLevel(logging.WARNING)  # Reduce log noise
        self.progress_bar = tqdm(total=len(self.search_results), desc="Scraping websites", unit="site")
        self.completed = 0

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(SearchSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(spider.item_scraped, signal=signals.item_scraped)
        crawler.signals.connect(spider.item_error, signal=signals.spider_error)
        return spider
    
    def item_scraped(self, item=None, response=None, spider=None):
        self.completed += 1
        self.progress_bar.update(1)
    
    def item_error(self, failure, response, spider):
        self.completed += 1
        self.progress_bar.update(1)
        
    def start_requests(self):
        for result in self.search_results:
            yield Request(
                url=result['href'],
                callback=self.parse,
                errback=self.handle_error,
                meta={'title': result['title']},
                dont_filter=True
            )
    
    def parse(self, response):
        title = response.meta.get('title')
        
        # Extract text from multiple HTML elements, not just paragraphs
        content_parts = []
        
        # Get headings
        for heading in response.css('h1, h2, h3, h4, h5, h6'):
            content_parts.append(heading.css('::text').get('').strip())
        
        # Get paragraphs with all nested text
        for p in response.css('p'):
            content_parts.append(' '.join(p.css('*::text').getall()).strip())
        
        # Get lists
        for li in response.css('li'):
            content_parts.append('• ' + ' '.join(li.css('*::text').getall()).strip())
        
        # Get div content that might contain text
        for div in response.css('div.content, div.article, article, section, main'):
            # Avoid duplicating already captured text
            div_text = ' '.join(div.css('::text').getall()).strip()
            if div_text and div_text not in content_parts:
                content_parts.append(div_text)
        
        # Filter out empty strings and join with single newlines
        content = '\n\n'.join(part.strip() for part in content_parts if part.strip())
        
        # Store content with clean formatting
        self.parsed_content[title] = f"\nFrom {title}:\n{content}\n"
        
        # Create a dummy item to satisfy the signal requirements
        dummy_item = {}
        
        # Signal that item has been scraped (for progress bar)
        self.crawler.signals.send_catch_log(
            signal=signals.item_scraped, 
            item=dummy_item,
            response=response, 
            spider=self
        )
    
    def handle_error(self, failure):
        request = failure.request
        title = request.meta.get('title')
        self.logger.warning(f"Error fetching {request.url} for '{title}': {repr(failure)}")
        
        # Signal error for progress bar
        self.crawler.signals.send_catch_log(signal=signals.spider_error, 
                                           failure=failure, response=None, spider=self)
    
    def spider_closed(self, spider):
        self.progress_bar.close()

def search_and_parse(query, max_results=5):
    # Show a progress bar for the search
    print(f"Searching for: {query}")
    search_progress = tqdm(total=1, desc="Retrieving search results", unit="query")
    
    # Get search results from DuckDuckGo
    results = ddg().text(query, max_results=max_results)
    search_progress.update(1)
    search_progress.close()
    
    if not results:
        print("No search results found.")
        return {"search_results": [], "parsed_content": ""}
    
    print(f"Found {len(results)} results. Starting content extraction...")
    
    # Create a container to store spider results
    spider_results = {}
    
    # Define a callback function to get the results when the spider closes
    def spider_closed(spider):
        spider_results['parsed_content'] = spider.parsed_content
    
    # Register the callback to the spider_closed signal
    dispatcher.connect(spider_closed, signal=signals.spider_closed)
    
    # Configure crawler settings
    settings = get_project_settings()
    settings.update({
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'CONCURRENT_REQUESTS': 16,
        'DOWNLOAD_TIMEOUT': 15,
        'RETRY_ENABLED': True,
        'RETRY_TIMES': 2,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        'LOG_LEVEL': 'ERROR',
    })
    
    # Set up the crawler process
    process = CrawlerProcess(settings)
    
    # Pass the spider class and its arguments separately
    process.crawl(SearchSpider, search_results=results)
    process.start()  # This blocks until the crawl is finished
    
    # Clean up the combined content
    combined_content = ''.join(spider_results.get('parsed_content', {}).values())
    
    # Apply HTML cleanup
    cleaned_content = clean_html_content(combined_content)
    
    return {
        "search_results": results,
        "parsed_content": cleaned_content
    }

def clean_html_content(text):
    # Handle HTML entities
    text = html.unescape(text)
    
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove strange Unicode characters that might be from HTML
    text = re.sub(r'[\u2028\u2029\ufeff]', ' ', text)
    
    # Fix common formatting issues
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    
    # Handle line breaks properly
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:  # Keep non-empty lines
            cleaned_lines.append(line)
    
    # Join with proper spacing
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Fix bullet points that might have been messed up
    cleaned_text = re.sub(r'• +', '• ', cleaned_text)
    
    return cleaned_text

if __name__ == "__main__":
    query = input("Enter your search query: ")
    result = search_and_parse(query)
    
    # Display search results
    print("\nSEARCH RESULTS:")
    for r in result["search_results"]:
        print(f"Title: {r['title']}")
        print(f"URL: {r['href']}")
        print(f"Snippet: {r['body']}")
        print("-" * 20)
    
    # Display parsed content
    print("\nPARSED CONTENT FROM TOP LINKS:")
    print(result["parsed_content"][-100:])  # Uncommented this line to actually show the content