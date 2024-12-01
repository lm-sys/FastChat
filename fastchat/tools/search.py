import os
import requests
from serpapi import GoogleSearch
from firecrawl import FirecrawlApp
from typing import List, Dict, Any

os.environ["YDC_API_KEY"] = ""
os.environ["FIRECRAWL_API_KEY"] = ""
os.environ["SERP_API_KEY"] = ""

def search_results_you(query: str, topk: int) -> List[Dict[str, Any]]:
    api_key = os.getenv("YDC_API_KEY")
    headers = {
        "X-API-Key": api_key,
        "Content-type": "application/json; charset=UTF-8",
    }
    params = {"query": query, "num_web_results": int(topk)}
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


def search_results_serp(query: str, topk: int) -> List[Dict[str, Any]]:
    print("searching SERP")
    params = {
        "q": query,
        "api_key": os.getenv("SERP_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    search_results = results['organic_results'][:topk]
    formatted_results = [
        {
            "title": result['title'],
            "url": result['link'],
            "text": result['snippet']
        }
        for result in search_results
    ]
    return formatted_results


def scrape_url(url: str) -> str:
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    response = app.scrape_url(url=url, params={'formats': ['markdown']})
    return response['markdown']


def formulate_web_summary(results: List[Dict[str, Any]], query: str, topk: int = 3) -> str:
    search_summary = f"Here are the summary of top {topk} search results for '{query}':\n"
    for result in results:
        search_summary += f"- [{result['title']}]({result['url']})\n"
        # add the snippets to the summary
        search_summary += f"Description: {result['text']}\n"
    return search_summary

def web_search(key_words: str, topk: int) -> str:
    results = search_results_serp(key_words, topk)
    # We only display the titles and urls in the search display
    search_display = "\n".join([f"- [{result['title']}]({result['url']})" for result in results])
    # We will store the search summary to the LLM context window
    search_summary = formulate_web_summary(results, key_words, topk)
    # We will scrape the content of the top search results for the very single-turn LLM response
    scraped_results = "\n".join([f"Title: {result['title']}:\n{scrape_url(result['url'])}\n" for result in results])
    return search_display, search_summary, scraped_results
