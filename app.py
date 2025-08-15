# --------------------------------------------------------------------------
# --- CRITICAL FIX for ChromaDB on Streamlit Cloud/Render ---
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
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import pytesseract
from PIL import Image
from pathlib import Path
from langchain_nomic import NomicEmbeddings

import requests
from bs4 import BeautifulSoup
import json

# --- FIX for Render Tesseract ---
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ----------------------------
# Load API Keys & Instantiate Models
# ----------------------------
load_dotenv()
groq_api_key = os.getenv("groq_apikey")
nomic_api_key = os.getenv("nomic_api") # Corrected variable name
scrapedo_api_key = os.getenv("SCRAPEDO_API_KEY")

if not all([groq_api_key, nomic_api_key, scrapedo_api_key]):
    st.error("API keys for Groq, Nomic, and Scrape.do must be set in your environment secrets.")
    st.stop()

llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
embeddings_model = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key=nomic_api_key)

# ----------------------------
# HELPER FUNCTION 1: Extract YouTube Info
# ----------------------------
def extract_youtube_info(text):
    url_match = re.search(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[a-zA-Z0-9_-]{11}\S*)", text)
    if not url_match: return None, None
    url = url_match.group(1)
    id_match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
    vid_id = id_match.group(1) if id_match else None
    return url, vid_id

# ----------------------------------------------------------------------------------
# --- REWRITTEN YOUTUBE FETCHER: The Definitive Middleman Scraper for TubeTranscript.com ---
# --- Based on your flawless HTML analysis ---
# ----------------------------------------------------------------------------------
def fetch_youtube_transcript(video_id: str, youtube_url: str) -> str | None:
    st.info("🚀 Using professional scraping service (Scrape.do) to automate tubetranscript.com...")
    
    # --- This is the new, correct plan based on your evidence ---
    # The site navigates to a new page, so a simple click script won't work.
    # We will simulate the form submission by constructing the results URL ourselves.
    # This is faster and more reliable.
    results_page_url = f"https://www.tubetranscript.com/en/watch?v={video_id}"
    
    st.info(f"Navigating directly to results page: {results_page_url}")

    params = {
        'token': scrapedo_api_key,
        'url': results_page_url,
        'render': 'false', # We don't need a full browser, just the raw HTML
    }
    
    try:
        api_url = "https://api.scrape.do/"
        response = requests.get(api_url, params=params, timeout=120)
        response.raise_for_status()

        # Scrape.do sends back the HTML of the results page
        soup = BeautifulSoup(response.text, 'lxml')
        
        # --- Find the hidden textarea with the golden ticket ID ---
        transcript_textarea = soup.find('textarea', id='restorealworkTranscriptData')
        
        if not transcript_textarea:
            st.error("Scraping succeeded, but could not find the hidden transcript textarea ('#restorealworkTranscriptData'). The site's layout may have changed.")
            return None
            
        transcript_text = transcript_textarea.get_text(strip=True)
        
        if transcript_text and len(transcript_text) > 50:
            st.success("✅ Success! Transcript fetched by scraping tubetranscript.com.")
            return transcript_text
        else:
            st.warning("Scraping succeeded, but the transcript was empty.")
            return None

    except Exception as e:
        st.error(f"An unexpected error occurred during automation: {e}")
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
    original_query = query 
    
    yt_url, yt_id = extract_youtube_info(query)
    if yt_url and yt_id:
        with st.spinner("Attempting to fetch YouTube transcript..."):
            yt_text = fetch_youtube_transcript(yt_id, yt_url) # Pass both id and url
            if yt_text:
                external_docs.append(Document(page_content=yt_text, metadata={"source": "YouTube"}))
                query = query.replace(yt_url, "").strip()

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
        with st.spinner("Analyzing content and preparing answer..."):
            summarization_keywords = ["summarize", "summary", "overview", "key points", "tldr", "main ideas"]
            is_summary_request = any(keyword in query.lower() for keyword in summarization_keywords) or not query

            splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
            chunks = splitter.split_documents(external_docs)

            if is_summary_request:
                st.info("Summarization request detected. Generating summary...")
                summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
                summary = summary_chain.run(chunks)
                st.write(summary)
            else:
                st.info("Specific question detected. Searching for the answer...")
                vs = Chroma.from_documents(chunks, embeddings_model)
                retriever = vs.as_retriever()
                prompt = ChatPromptTemplate.from_template("Answer the user's question based only on the provided context.\n\n<context>{context}</context>\n\nQuestion: {input}")
                doc_chain = create_stuff_documents_chain(llm, prompt)
                chain = create_retrieval_chain(retriever, doc_chain)
                resp = chain.invoke({"input": query})
                st.write(resp["answer"])
    
    elif not yt_id:
        st.write("Answering general question...")
        st.write(llm.invoke(original_query).content)
