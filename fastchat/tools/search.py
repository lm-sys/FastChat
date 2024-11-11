import os
import requests
from firecrawl import FirecrawlApp
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv('keys.env')

def search_results_you(query: str, topk: int) -> List[Dict[str, Any]]:
    api_key = os.getenv("YDC_API_KEY")
    headers = {
        "X-API-Key": api_key,
        "Content-type": "application/json; charset=UTF-8",
    }
    params = params = {"query": query, "num_web_results": topk}
    response = requests.get(f"https://api.ydc-index.io/search", params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(f"You.com API returned error code {response.status_code} - {response.reason}")
    data = response.json()
    hits = data.get("hits", [])
    formatted_results = [
        {
            "title": hit["title"],
            "url": hit["url"],
            "text": "\n".join(hit.get("snippets", []))
        }
        for hit in hits
    ]
    return formatted_results

def scrape_url(url: str) -> str:
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    response = app.scrape_url(url=url, params={'formats': ['markdown']})
    return response['markdown']

def web_search(query: str, topk: int) -> str:
    results = search_results_you(query, topk)
    scraped_results = [f"Title: {result['title']}:\n{scrape_url(result['url'])}\n" for result in results]
    return "\n".join(scraped_results)
        