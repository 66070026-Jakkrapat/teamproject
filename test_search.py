import asyncio
from playwright.async_api import async_playwright
from urllib.parse import quote_plus

async def test_search():
    keyword = "เทรนด์ธุรกิจ SME 2025"
    print(f"Searching for: {keyword}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        url = f"https://www.google.com/search?q={quote_plus(keyword)}"
        print(f"URL: {url}")
        
        await page.goto(url, wait_until="networkidle", timeout=30000)
        
        # Method 1: All links
        all_links = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
        print(f"Total links found: {len(all_links)}")
        
        # Method 2: Main search results (headers)
        cite_links = await page.eval_on_selector_all("cite", "els => els.map(e => e.closest('a')?.href).filter(Boolean)")
        print(f"Cite links: {cite_links}")
        
        await page.screenshot(path="google_test.png")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_search())
