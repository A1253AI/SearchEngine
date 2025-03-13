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

def split_text_into_chunks(text, chunk_size):
   Split text into chunks of approximately equal size
   Input: A long block of text and a chunk_size (usually in characters).
   Output: A list of chunks, each containing complete sentences and approximately chunk_size characters.
   Logic:
   Splits the text into sentences using a regular expression that preserves sentence-ending punctuation.
   Iteratively adds sentences to a chunk until adding the next one would exceed the target chunk size.
   Starts a new chunk and repeats.


def process_text_content(texts, chunk_size)
    Uses asyncio.get_event_loop() + run_in_executor to:
    Run split_text_into_chunks() concurrently for each text input.
    It splits the text at each sentence (by . ).
    It adds sentences to the current chunk until adding another one would exceed the chunk_size.
    Then it starts a new chunk.

def get_embeddings_from_ollama(text_chunks) - embedding text chunks using
   Sends each chunk as a POST request to your local Ollama API for embeddings.
   Handles errors by adding dummy embeddings ([0.0] * 768) to maintain alignment.
   Uses httpx.AsyncClient for efficient async HTTP requests.
   It calls get_embeddings_from_ollama() to get embeddings from the local Ollama model (nomic-embed-text).
   ex - {
  "model": "nomic-embed-text",
  "prompt": "we will win."
      }
  Output:
  You get back an embedding, which is a list of float numbers (e.g, a 768-dimensional vector):
  [
       0.00234, -0.0317, 0.1075, ..., 0.0541
  ]
   
def query_embeddings(text)


Generates an embedding vector for a user’s query.

How it works:

Wraps the query string in a list and uses get_embeddings_from_ollama.
Returns the single embedding.

def query_ollama_llm(prompt)
Purpose:
Queries Ollama's local LLM llama 3.2 to generate a response based on a prompt.

How it works:

Sends POST request to /api/generate with prompt.


def get_embeddings_from_ollama(text_chunks)
Purpose:
Gets vector embeddings for each text chunk using Ollama’s local embedding model (nomic-embed-text).

How it works:

For each chunk:
If non-empty, sends a POST request to Ollama’s /api/embeddings.
Appends the embedding or a zero-vector if it fails.







