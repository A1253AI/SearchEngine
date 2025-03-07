from urllib.parse import unquote
import streamlit as st
import aiohttp
import asyncio
import nest_asyncio
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import requests
import re
import pandas as pd
import faiss
import numpy as np
import httpx
import os
import random
import json

# Apply the nest_asyncio patch
nest_asyncio.apply()

# Configuration
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1"
}

total_results_to_fetch = 10
chunk_size = 1000

# Define output paths 
output_dir = "search_data"
os.makedirs(output_dir, exist_ok=True)
dataframe_out_path = os.path.join(output_dir, "search_data.csv")
faiss_index_path = os.path.join(output_dir, "faiss_index.index")

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama3.2:1b" # LLM model name

async def fetch(session, url, params=None):
    """Fetch content from a URL with optional parameters."""
    try:
        async with session.get(url, params=params, headers=headers, timeout=30) as response:
            return await response.text()
    except Exception as e:
        st.error(f"Error fetching URL {url}: {str(e)}")
        return ""

async def fetch_page_bing(session, query, page_num, results):
    """Fetch search results from Bing"""
    try:
        offset = (page_num - 1) * 10
        bing_url = "https://www.bing.com/search"
        params = {"q": query, "first": offset}
        
        html = await fetch(session, bing_url, params)
        
        if not html:
            return
            
        soup = BeautifulSoup(html, 'html.parser')
            
        result_blocks = soup.select('li.b_algo')
        
        for block in result_blocks:
            title_elem = block.select_one('h2 a')
            if not title_elem:
                continue
                
            title = title_elem.text.strip()
            url = title_elem.get('href', '')
            
            if url and (url.startswith('http://') or url.startswith('https://')):
                results.append({"title": title, "links": url})
                
            if len(results) >= total_results_to_fetch:
                break

    except Exception as e:
        st.error(f"Error parsing Bing search results: {str(e)}")

async def fetch_page_ddg(session, query, page_num, results):
    """Fetch search results from DuckDuckGo"""
    try:
        ddg_url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        
        html = await fetch(session, ddg_url, params)
        
        if not html:
            return
            
        soup = BeautifulSoup(html, 'html.parser')
            
        result_blocks = soup.select('.result')
        
        for block in result_blocks:
            title_elem = block.select_one('.result__title a')
            if not title_elem:
                continue
                
            title = title_elem.text.strip()
            url = title_elem.get('href', '')
            
            if url:
                url_match = re.search(r'uddg=([^&]+)', url)
                if url_match:
                    url = unquote(url_match.group(1))
                    
            if url and (url.startswith('http://') or url.startswith('https://')):
                results.append({"title": title, "links": url})
                
            if len(results) >= total_results_to_fetch:
                break

    except Exception as e:
        st.error(f"Error parsing DuckDuckGo search results: {str(e)}")

async def fetch_page(session, query, page_num, results):
    """Fetch results from selected search engines"""
    try:
        search_engine = st.session_state.get("search_engine", "DuckDuckGo")
        
        if search_engine == "Bing":
            await fetch_page_bing(session, query, page_num, results)
        else:
            await fetch_page_ddg(session, query, page_num, results)
            
        if not results and search_engine != "Bing":
            await fetch_page_bing(session, query, page_num, results)

    except Exception as e:
        st.error(f"Error in search results fetching: {str(e)}")

def get_all_text_from_url(url):
    """Improved URL content fetching"""
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return ""
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for element in soup.select('script, style, nav, footer, header, aside, iframe'):
            element.decompose()
            
        text = soup.get_text(separator='\n', strip=True)
        return ' '.join(text.split()).strip()
        
    except Exception as e:
        return ""

def split_text_into_chunks(text, chunk_size):
    """Split text into chunks with overlap."""
    if not text or len(text) < 50:
        return []
        
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-2:] + [sentence]
            current_length = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    return chunks if chunks else [text]

async def get_embeddings_from_ollama(text_chunks):
    """Get embeddings using Ollama"""
    embeddings = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for chunk in text_chunks:
            if not chunk.strip():
                embeddings.append(np.random.uniform(-0.01, 0.01, 768).tolist())
                continue
                
            try:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": OLLAMA_EMBED_MODEL, "prompt": chunk[:5000]}
                )
                if response.status_code == 200:
                    data = response.json()
                    embeddings.append(data.get("embedding", []))
                else:
                    embeddings.append(np.random.uniform(-0.01, 0.01, 768).tolist())
            except Exception:
                embeddings.append(np.random.uniform(-0.01, 0.01, 768).tolist())
    
    return embeddings

async def generate_answer_with_ollama(query, context_chunks):
    """Generate answer using Ollama's LLM"""
    if not context_chunks:
        return "No relevant information found."
    
    combined_chunks = " ".join(context_chunks)[:12000]
    
    prompt = f"""Answer this question based on the context:
    Question: {query}
    Context: {combined_chunks}"""
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False}
            )
            return response.json().get("response", "Unable to generate answer.")
    except Exception:
        return "Error generating answer."

async def fetch_and_process_data(search_query):
    """Process search query and create index"""
    try:
        async with aiohttp.ClientSession() as session:
            results = []
            tasks = [fetch_page(session, search_query, page_num, results) for page_num in range(1, 4)]
            await asyncio.gather(*tasks)
        
        if not results:
            st.error("No search results found.")
            return None
            
        urls = [result['links'] for result in results]
        with ThreadPoolExecutor(max_workers=10) as executor:
            loop = asyncio.get_event_loop()
            texts = await asyncio.gather(*[loop.run_in_executor(executor, get_all_text_from_url, url) for url in urls])
        
        data = []
        for result, text in zip(results, texts):
            if not text.strip():
                continue
                
            chunks = split_text_into_chunks(text, chunk_size)
            if not chunks:
                continue
                
            embeddings = await get_embeddings_from_ollama(chunks)
            for chunk, embedding in zip(chunks, embeddings):
                data.append({
                    'title': result['title'],
                    'url': result['links'],
                    'chunk': chunk,
                    'embedding': embedding
                })
        
        if not data:
            st.error("No processable content found.")
            return None
            
        df = pd.DataFrame(data).drop(columns=['embedding'])
        df.to_csv(dataframe_out_path, index=False)
        
        embeddings = np.array([entry['embedding'] for entry in data], dtype=np.float32)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, faiss_index_path)
        
        return data
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None


#  query_vector_store function
def query_vector_store(query_embedding, k=5):

    """Query the FAISS index and return top-k unique results"""
    try:
        # Load FAISS index
        index = faiss.read_index(faiss_index_path)
        
        # Ensure query embedding is in the correct format
        query_embedding = np.array([query_embedding], dtype=np.float32)

        # Perform search
        distances, indices = index.search(query_embedding, k)

        # Load the associated DataFrame
        df = pd.read_csv(dataframe_out_path)

        results = []
        seen_titles = set()  # Track seen titles to avoid duplicates

        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(df):  # Ensure valid index
                title = df.iloc[idx]['title']
                if title not in seen_titles:
                    seen_titles.add(title)
                    results.append({
                        'title': title,
                        'url': df.iloc[idx]['url'],
                        'chunk': df.iloc[idx]['chunk'],
                        'score': float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
                    })

        # Sort results by score (higher is better)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    except Exception as e:
        print(f"Error querying FAISS index: {e}")
        return []
# Streamlit UI
st.title("Search Engine")

# Search engine selection
search_engine = st.radio(
    "Select search engine:",
    ["DuckDuckGo", "Bing"],
    horizontal=True
)
st.session_state["search_engine"] = search_engine

query = st.text_input("Enter your search query:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a search query")
        st.stop()
        
    with st.spinner(f"Searching with {search_engine}..."):
        processed_data = asyncio.run(fetch_and_process_data(query))
        
        if not processed_data:
            st.stop()
            
        query_embedding = asyncio.run(get_embeddings_from_ollama([query]))[0]
        results = query_vector_store(query_embedding)
        
        if not results:
            st.warning("No relevant results found.")
            st.stop()
            
        answer = asyncio.run(generate_answer_with_ollama(query, [r['chunk'] for r in results]))
        
        st.subheader("Answer")
        st.write(answer)
        
        st.subheader("Top Results")
        for i, result in enumerate(results, 1):
            st.markdown(f"**{i}. {result['title']}**")
            st.markdown(f"*Score: {result['score']:.2f}*")
            st.markdown(f"[{result['url']}]({result['url']})")
            st.write(result['chunk'][:300] + "...")
            st.write("---")
