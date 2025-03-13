# SearchEngine
# AI powered Search Engine

Developed an intelligent search engine leveraging local LLMs for semantic understanding and embeddings. Designed a FAISS-based vector store for efficient similarity search and ranking. Implemented web scraping (Bing, DuckDuckGo) with advanced parsing and error handling for robust data collection. Built a Streamlit-based UI with paginated results, dynamic answer generation, and interactive exploration. Integrated model selection, connection monitoring, and chunk-based text processing for large documents. Developed context-aware response generation and relevance-based ranking with similarity scoring. Added error handling and fallback mechanisms for seamless operation across various search scenarios.

Added a User-Agent that mimics a real browser,reducing the chance of being blocked or flagged.

async def fetch(session, url, params=None):
      Makes an asynchronous HTTP GET request using an aiohttp session.
      Includes:
      params: query parameters for the URL (like q=search+term)
      headers: to simulate a browser 
      timeout: sets max wait time for response (30 seconds)
      async -  Allows multiple web requests to run concurrently (non-blocking), which speeds up scraping multiple pages.

  async def fetch_page()
     Calls the fetch() function to get one BROWSER search results page (e.g., page 1, 2, 3…).
     Parses the page HTML using BeautifulSoup.
     Extracts:
     .tF2Cxc: wrapper for each result
      .DKV0Md: result title
     .yuRUbf a: link to the result


  def get_all_text_from_url(url):
    Extract and clean all text content from a URL->it fetches and extracts readable text from a webpage effectively.
    Sends a GET request to the given url with headers to mimic a browser.
    Parses the HTML using BeautifulSoup.
    Removes unwanted tags (<script>, <style>) that don't contain useful content.
    Extracts visible text, removes extra whitespace.
    Returns a clean, human-readable string of text.
