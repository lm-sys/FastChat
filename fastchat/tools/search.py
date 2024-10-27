import requests

YOU_SEARCH_API_KEY = "YOUR API KEY"

def get_ai_snippets_for_query(query, num_web_results=1):
    headers = {"X-API-Key": YOU_SEARCH_API_KEY}
    params = {"query": query, "num_web_results": num_web_results}
    return requests.get(
        f"https://api.ydc-index.io/search",
        params=params,
        headers=headers,
    ).json()

def format_search_results(results):
    formatted_results = ''
    results = results['hits']
    for result in results:
        formatted_results += result['url'] + '\n'
        formatted_results += result['title'] + '\n'
        for snippet in result['snippets']:
            formatted_results += snippet + '\n'
            formatted_results += '--------------------------------\n'
        formatted_results += '--------------------------------\n'
    return formatted_results

def web_search(key_words, topk=1):
    web_search_results = get_ai_snippets_for_query(query=key_words, num_web_results=topk)
    web_search_results = format_search_results(web_search_results)
    return web_search_results