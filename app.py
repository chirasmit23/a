# --------------------------------------------------------------------------
# --- CRITICAL FIX for ChromaDB on Streamlit Cloud/Render ---
# This "patch" must be at the very top of the file, before any other imports
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --------------------------------------------------------------------------

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

# --- NEW IMPORTS for the Professional Scraping API ---
import requests
from bs4 import BeautifulSoup
import json # To create the action script

# --- FIX for Render Tesseract ---
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ----------------------------
# Load API Keys & Instantiate Models
# ----------------------------
load_dotenv()
groq_api_key = os.getenv("groq_apikey")
# --- FIX: Corrected your variable name to the standard to avoid future errors ---
nomic_api_key = os.getenv("nomic_api")
scrapedo_api_key = os.getenv("SCRAPEDO_API_KEY")

if not all([groq_api_key, nomic_api_key, scrapedo_api_key]):
    st.error("API keys for Groq, Nomic, and Scrape.do must be set in your environment secrets.")
    st.stop()

llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
embeddings_model = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key=nomic_api_key)

# ----------------------------
# Extract YouTube ID & Link (Unchanged)
# ----------------------------
def extract_youtube_info(text):
    url_match = re.search(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[a-zA-Z0-9_-]{11}\S*)", text)
    if not url_match: return None, None
    url = url_match.group(1)
    id_match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
    vid_id = id_match.group(1) if id_match else None
    return url, vid_id

# ----------------------------------------------------------------------------------
# --- The Professional "Remote Control" Transcript Fetcher with a Human Pause ---
# ----------------------------------------------------------------------------------
def fetch_transcript_with_remote_actions(youtube_url: str) -> str | None:
    """
    Uses the Scrape.do API to remotely control a browser, performing the
    exact steps a human would: type, PAUSE, click, wait, and extract.
    """
    st.info("🚀 Using professional scraping service (Scrape.do) with remote browser actions...")
    
    target_site_url = "https://www.youtube-transcript.io/"
    
    action_script = [
        {"Action": "Fill", "Selector": "#youtube-url-input", "Value": youtube_url},
        {"Action": "Wait", "Timeout": 2000},
        {"Action": "Click", "Selector": "form button[type=\"submit\"]"},
        {"Action": "WaitSelector", "WaitSelector": "main p", "Timeout": 25000}
    ]

    actions_json_string = json.dumps(action_script)

    params = {
        'token': scrapedo_api_key,
        'url': target_site_url,
        'render': 'true',
        'playWithBrowser': actions_json_string
    }
    
    try:
        api_url = "https://api.scrape.do/"
        response = requests.get(api_url, params=params, timeout=120)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'lxml')
        main_content = soup.find('main')
        if not main_content:
            st.error("Remote browser failed to find the main content area after actions.")
            return None

        paragraphs = main_content.find_all('p')
        if not paragraphs:
            st.warning("Remote browser actions completed, but no transcript paragraphs were found.")
            return None
            
        transcript_text = " ".join(p.get_text(strip=True) for p in paragraphs)
        
        if transcript_text and len(transcript_text) > 50:
            st.success("✅ Success! Transcript fetched via remote browser automation.")
            return transcript_text
        else:
            st.warning("Remote browser ran, but the transcript was empty.")
            return None

    except requests.exceptions.RequestException as e:
        if e.response is not None and e.response.status_code == 400:
            st.error("The API request was rejected as a 'Bad Request'. The action script or parameters may be invalid.")
        else:
            st.error(f"Failed to connect to Scrape.do API or the request timed out. Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during remote automation: {e}")
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
    if yt_url and yt_id:
        with st.spinner("Attempting to fetch YouTube transcript using remote browser automation..."):
            yt_text = fetch_transcript_with_remote_actions(yt_url)
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
