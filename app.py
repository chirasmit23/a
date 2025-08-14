import os
import re
import webvtt
import subprocess
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
import pytesseract
from PIL import Image
from youtube_transcript_api import YouTubeTranscriptApi
from pathlib import Path

# --- NEW: Import Nomic Embeddings ---
from langchain_nomic import NomicEmbeddings

# ----------------------------
# Load API Keys
# ----------------------------
load_dotenv()
groq_api_key = os.getenv("groq_apikey")
nomic_api_key = os.getenv("nomic_api") # Nomic's API key

# ----------------------------
# --- NEW: Instantiate Embedding Model ---
# ----------------------------
# We initialize the embedding model once. LangChain will handle calling it.
# This replaces the entire hf_client and embed_texts function.
if not nomic_api_key:
    st.error("NOMIC_API_KEY not found in .env file. Please add it to continue.")
    st.stop()
    
embeddings_model = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key=nomic_api_key)

# ----------------------------
# Extract YouTube ID & Link
# ----------------------------
def extract_youtube_info(text):
    """Extracts YouTube URL and Video ID from a given text string."""
    url_match = re.search(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[a-zA-Z0-9_-]{11}\S*)", text)
    if not url_match: return None, None
    url = url_match.group(1)
    id_match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
    vid_id = id_match.group(1) if id_match else None
    return url, vid_id

# ----------------------------
# Stealthy YouTube Transcript Fetcher
# ----------------------------
def fetch_youtube_transcript(video_id, url):
    """
    Fetches a YouTube transcript, first trying the fast API, then falling back
    to a stealthier yt-dlp method designed to look like a human browser.
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'hi'])
        transcript = " ".join(seg["text"] for seg in transcript_list if seg["text"].strip())
        if transcript:
            st.info("✅ Transcript fetched quickly via API.")
            return transcript
    except Exception:
        pass

    st.info("🏃‍♂️ API method failed, falling back to stealth mode... (this may take a moment)")
    
    if not shutil.which("yt-dlp"):
        st.error("yt-dlp command not found. Please ensure it's installed.")
        return None

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Referer': url,
    }
    header_args = [arg for key, value in headers.items() for arg in ['--add-header', f'{key}: {value}']]
    output_template = f"{video_id}.%(ext)s"
    vtt_path = f"{video_id}.en.vtt"

    cmd = ["yt-dlp", "--skip-download", "--write-auto-sub", "--sub-lang", "en", "--sub-format", "vtt", "-o", output_template, *header_args, url]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        if not os.path.exists(vtt_path): return None
        transcript_text = "\n".join(caption.text for caption in webvtt.read(vtt_path))
        st.info("✅ Transcript fetched successfully in stealth mode.")
        return transcript_text.strip() or None
    except subprocess.TimeoutExpired:
        st.error("yt-dlp command timed out.")
        return None
    except subprocess.CalledProcessError as e:
        if "confirm you’re not a bot" in e.stderr:
            st.error("❌ YouTube is blocking automated requests with a bot-check.")
        else:
            st.error(f"An error occurred with yt-dlp:\n```\n{e.stderr}\n```")
        return None
    finally:
        if os.path.exists(vtt_path): os.remove(vtt_path)

# ----------------------------
# Build vector store for permanent PDFs (optional)
# ----------------------------
@st.cache_resource
def build_custom_vectorstore():
    PDF_PATHS = []
    if not PDF_PATHS: return None
    docs = [doc for p in PDF_PATHS for doc in PyPDFLoader(p).load()]
    if not docs: return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    # --- SIMPLIFIED: LangChain handles the embedding process directly ---
    return FAISS.from_documents(chunks, embeddings_model)

custom_vs = build_custom_vectorstore()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Multi-Source Q&A Bot", layout="wide")
st.title("📄 PDF, 🖼️ Image & 🎥 YouTube Q&A Bot")

uploads = st.file_uploader("Upload PDFs or Images", type=["pdf","png","jpg","jpeg","webp"], accept_multiple_files=True)
query = st.text_input("Ask a question or paste a YouTube link", key="query_input").strip()

if query:
    external_docs = []
    
    yt_url, yt_id = extract_youtube_info(query)
    if yt_id:
        with st.spinner("Fetching YouTube transcript..."):
            yt_text = fetch_youtube_transcript(yt_id, yt_url)
            if yt_text:
                external_docs.append(Document(page_content=yt_text, metadata={"source": "YouTube"}))
                query = query.replace(yt_url, "").strip() or "Summarize the content of the video."

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
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(external_docs)
            
            # --- SIMPLIFIED: Directly create vector store from documents ---
            # LangChain now handles the call to Nomic's API automatically.
            vs = FAISS.from_documents(chunks, embeddings_model)
            
            prompt = ChatPromptTemplate.from_template("Answer based only on the context below.\n<context>{context}</context>\nQuestion: {input}")
            llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
            doc_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vs.as_retriever()
            chain = create_retrieval_chain(retriever, doc_chain)
            
            resp = chain.invoke({"input": query})
            st.write(resp["answer"])
    
    elif not yt_id:
        llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
        # Simplified general chat logic
        if custom_vs and "i don't know" not in (ans := create_retrieval_chain(custom_vs.as_retriever(), create_stuff_documents_chain(llm, ...)).invoke({"input": query}).get("answer", "")):
            st.write(ans)
        else:
            st.write(llm.invoke(query).content)
