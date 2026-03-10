import asyncio
from backend.services.scraper import fetch_google_search_results

async def main():
    keyword = "เทรนด์ธุรกิจ SME 2025"
    print(f"Searching Google for: '{keyword}'")
    urls = await fetch_google_search_results(keyword, max_results=3)
    print("Found URLs:")
    for i, url in enumerate(urls, 1):
        print(f"{i}. {url}")

if __name__ == "__main__":
    asyncio.run(main())
