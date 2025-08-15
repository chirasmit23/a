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

# --- NEW IMPORTS for the Professional Solution ---
import subprocess
import shutil
import webvtt

# --- FIX for Render Tesseract ---
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ----------------------------
# Load API Keys & Instantiate Models
# ----------------------------
load_dotenv()
groq_api_key = os.getenv("groq_apikey")
nomic_api_key = os.getenv("nomic_api")

if not groq_api_key or not nomic_api_key:
    st.error("API keys for Groq and Nomic must be set in your environment secrets.")
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

# --------------------------------------------------------------------------
# --- REWRITTEN YOUTUBE FETCHER: The Ultimate Professional Method ---
# --- yt-dlp + Scrape.do Proxy Mode ---
# --------------------------------------------------------------------------
def fetch_youtube_transcript(video_id, url):
    """
    Fetches a YouTube transcript using the most robust method available:
    yt-dlp combined with Scrape.do's Proxy Mode using a residential proxy.
    """
    proxy_username = st.secrets.get("SCRAPEDO_USERNAME") # Your API Token
    proxy_password = st.secrets.get("SCRAPEDO_PASSWORD") # Should be "super=true"
    proxy_host_port = st.secrets.get("SCRAPEDO_HOST_PORT") # Should be "proxy.scrape.do:8080"

    if not all([proxy_username, proxy_password, proxy_host_port]):
        st.error("❌ Scrape.do Proxy Mode credentials are not configured in secrets.")
        return None

    st.info("🚀 Using professional residential proxy to fetch transcript directly from YouTube...")
    proxy_url = f"http://{proxy_username}:{proxy_password}@{proxy_host_port}"
    
    output_template = f"{video_id}"
    vtt_path = f"{video_id}.en.vtt"

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--sub-format", "vtt",
        "-o", output_template,
        "--proxy", proxy_url, # The key to success
        url
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120) 
        if not os.path.exists(vtt_path):
            st.warning("Proxy succeeded, but yt-dlp did not find or produce a transcript file for this video.")
            return None
        transcript_text = "\n".join(caption.text for caption in webvtt.read(vtt_path))
        st.success("✅ Transcript fetched successfully using professional proxy!")
        return transcript_text.strip() or None
    except Exception as e:
        st.error(f"The proxy method failed. Check credentials/parameters. Error: {e}")
        return None
    finally:
        # Cleanup any files yt-dlp might create
        if os.path.exists(vtt_path):
            os.remove(vtt_path)
        if os.path.exists(f"{video_id}.info.json"):
            os.remove(f"{video_id}.info.json")

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
        with st.spinner("Fetching YouTube transcript..."):
            yt_text = fetch_youtube_transcript(yt_id, yt_url)
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
