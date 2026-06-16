import os
import requests

# The base URL for the NewsAPI "everything" endpoint.
# This endpoint searches the full text and titles of articles.
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# The search terms we care about — each one is sent as a
# separate request and the results are combined at the end.
SEARCH_QUERIES = [
    "acquisition",
    "acquired",
    "merger",
    "buyout",
    "purchased",
]


def fetch_headlines():
    """
    Query NewsAPI for each search term and return a deduplicated
    list of headline strings.
    """
    api_key = os.environ.get("NEWS_API_KEY", "")

    if not api_key:
        print("ERROR: NEWS_API_KEY is not set in your .env file.")
        return []

    all_headlines = []
    seen = set()

    for query in SEARCH_QUERIES:
        params = {
            "q":        query,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": 20,
            "apiKey":   api_key,
        }

        response = requests.get(NEWSAPI_URL, params=params)
        data = response.json()

        if data.get("status") != "ok":
            print(f"  NewsAPI error for '{query}': {data.get('message', 'unknown error')}")
            continue

        for article in data.get("articles", []):
            headline = article.get("title", "").strip()

            if not headline:
                continue
            if headline in seen:
                continue

            seen.add(headline)
            all_headlines.append(headline)

    print(f"Fetched {len(all_headlines)} unique headlines from NewsAPI.")
    return all_headlines
