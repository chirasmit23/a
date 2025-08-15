import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
import pytesseract
from PIL import Image
from pathlib import Path
from langchain_nomic import NomicEmbeddings

# --- NEW IMPORTS FOR THE SCRAPER GAUNTLET ---
import time
import random
import cloudscraper # A special library to bypass services like Cloudflare
from bs4 import BeautifulSoup

# --- FIX for Render Tesseract ---
# This line is crucial for Tesseract to work on Render's environment
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ----------------------------
# Load API Keys & Instantiate Models
# ----------------------------
load_dotenv()
groq_api_key = os.getenv("groq_apikey")
# --- FIX: Your previous code had a typo here, corrected to the standard name ---
nomic_api_key = os.getenv("nomic_api")

if not groq_api_key or not nomic_api_key:
    st.error("API keys for Groq and Nomic must be set in your environment secrets.")
    st.stop()

llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
embeddings_model = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key=nomic_api_key)

# ----------------------------
# Helper function: Extract YouTube Info (Unchanged)
# ----------------------------
def extract_youtube_info(text):
    url_match = re.search(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[a-zA-Z0-9_-]{11}\S*)", text)
    if not url_match: return None, None
    url = url_match.group(1)
    id_match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
    vid_id = id_match.group(1) if id_match else None
    return url, vid_id

# -------------------------------------------------------------
# --- NEW: THE FULL 5-STAGE "SCRAPER GAUNTLET" ---
# -------------------------------------------------------------
def fetch_transcript_via_scraping_gauntlet(video_id: str) -> str | None:
    """
    Attempts to scrape a YouTube transcript by running a gauntlet of third-party websites.
    """
    scraper = cloudscraper.create_scraper()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    }

    # --- We define a specific scraper function for each website's unique HTML ---
    def _scrape_youtubetotranscript(sid):
        url = f"https://youtubetotranscript.com/?v={sid}"
        st.info(f"Gauntlet Stage 1: Trying youtubetotranscript.com...")
        response = scraper.get(url, headers=headers, timeout=25)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        container = soup.find('div', class_='transcript-text')
        if not container: return None
        return " ".join(p.get_text(strip=True) for p in container.find_all('p'))

    def _scrape_youtubetranscript_io(sid):
        url = f"https://www.youtube-transcript.io/transcript/?v={sid}"
        st.info(f"Gauntlet Stage 2: Trying youtube-transcript.io...")
        response = scraper.get(url, headers=headers, timeout=25)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        paragraphs = soup.find_all('p')
        if not paragraphs: return None
        return " ".join(p.get_text(strip=True) for p in paragraphs)

    # The following sites are modern JavaScript apps and CANNOT be scraped this way.
    # These functions are designed to fail gracefully.
    def _scrape_tactiq(sid):
        st.info(f"Gauntlet Stage 3: Trying tactiq.io (Known to be unscrapable)...")
        return None

    def _scrape_notegpt(sid):
        st.info(f"Gauntlet Stage 4: Trying notegpt.io (Known to be unscrapable)...")
        return None

    def _scrape_komeai(sid):
        st.info(f"Gauntlet Stage 5: Trying kome.ai (Known to be unscrapable)...")
        return None
    
    # Run the Gauntlet in order of most likely to succeed
    scrapers_to_try = [
        _scrape_youtubetotranscript,
        _scrape_youtubetranscript_io,
        _scrape_tactiq,
        _scrape_notegpt,
        _scrape_komeai,
    ]
    
    for scrape_function in scrapers_to_try:
        try:
            time.sleep(random.uniform(2.0, 4.0)) # Human-like pause
            transcript = scrape_function(video_id)
            if transcript and len(transcript) > 100:
                # --- THIS IS THE SUCCESS MESSAGE YOU WANTED ---
                site_name = scrape_function.__name__.replace("_scrape_", "").replace("_", ".")
                st.success(f"✅ Web scraping successful! Fetched transcript from: {site_name}")
                return transcript
            else:
                st.warning(f"...{scrape_function.__name__} failed or returned empty content.")
        except Exception as e:
            st.warning(f"...{scrape_function.__name__} failed with an error: {str(e)[:100]}...")
            continue
            
    st.error("❌ All scraping attempts failed. The websites may be blocking us, have changed their layout, or the video has no transcript.")
    return None

# ----------------------------
# Streamlit UI & Main Logic
# ----------------------------
st.set_page_config(page_title="Multi-Source Q&A Bot", layout="wide")
st.title("📄 PDF, 🖼️ Image & 🎥 YouTube Q&A Bot")
st.write("Powered by Groq Llama3 & Nomic Embeddings")

uploads = st.file_uploader("Upload PDFs or Images", type=["pdf","png","jpg","jpeg","webp"], accept_multiple_files=True)
query = st.text_input("Ask a question or paste a YouTube link", key="query_input").strip()

if query:
    external_docs = []
    
    yt_url, yt_id = extract_youtube_info(query)
    if yt_id:
        with st.spinner("Attempting to fetch YouTube transcript via scraping gauntlet..."):
            # --- MODIFIED: The old fetch_youtube_transcript is gone. We now call the gauntlet. ---
            yt_text = fetch_transcript_via_scraping_gauntlet(yt_id)
            if yt_text:
                external_docs.append(Document(page_content=yt_text, metadata={"source": "YouTube"}))
                query = query.replace(yt_url, "").strip() or "Summarize the video."

    if uploads:
        upload_dir = Path("uploaded_files")
        upload_dir.mkdir(exist_ok=True)
        for f in uploads:
            path = upload_dir / f.name
            path.write_bytes(f.getbuffer())
            if f.name.lower().endswith(".pdf"):
                external_docs.extend(PyPDFLoader(str(path)).load())
            else:
                try:
                    text = pytesseract.image_to_string(Image.open(path), config="--psm 6").strip()
                    if text: external_docs.append(Document(page_content=text, metadata={"source": f.name}))
                except Exception as e:
                    st.warning(f"Could not process image {f.name}: {e}")

    if external_docs:
        with st.spinner("Embedding content and preparing answer..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(external_docs)
            vs = Chroma.from_documents(chunks, embeddings_model)
            prompt = ChatPromptTemplate.from_template("Answer the user's question based only on the provided context.\n\n<context>{context}</context>\n\nQuestion: {input}")
            doc_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vs.as_retriever()
            chain = create_retrieval_chain(retriever, doc_chain)
            resp = chain.invoke({"input": query})
            st.write(resp["answer"])
    
    elif not yt_id:
        st.write("Answering general question...")
        st.write(llm.invoke(query).content)
