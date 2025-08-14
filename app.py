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
from huggingface_hub import InferenceClient

# ----------------------------
# Load API Keys
# ----------------------------
load_dotenv()
groq_api_key = os.getenv("groq_apikey")
hf_api_key ="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"

# ----------------------------
# HF Embedding Function (API-based)
# ----------------------------
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Small & fast
hf_client = InferenceClient(api_key=hf_api_key)

def embed_texts(texts):
    embeddings = []
    for t in texts:
        response = hf_client.post(
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBED_MODEL}",
            json={"inputs": t}
        )
        embeddings.append(response[0])
    return embeddings

# ----------------------------
# Extract YouTube ID & Link
# ----------------------------
def extract_youtube_info(text):
    url_match = re.search(r"(https?://\S+)", text)
    url = url_match.group(1) if url_match else None
    id_match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url or "")
    vid_id = id_match.group(1) if id_match else None
    return url, vid_id

# ----------------------------
# Fetch transcript via API or yt-dlp + webvtt
# ----------------------------
def fetch_youtube_transcript(video_id, url):
    for lang in ("en", "hi"):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
            return " ".join(seg["text"] for seg in transcript if seg["text"].strip())
        except:
            pass
    if not shutil.which("yt-dlp"):
        return None
    vtt_path = f"{video_id}.en.vtt"
    cmd = [
        "yt-dlp", "--skip-download",
        "--write-auto-sub", "--sub-lang", "en",
        "--sub-format", "vtt",
        "-o", video_id,
        url
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if not os.path.exists(vtt_path):
            return None
        text = "\n".join(caption.text for caption in webvtt.read(vtt_path))
        os.remove(vtt_path)
        return text.strip() or None
    except subprocess.CalledProcessError as e:
        st.error(f"yt-dlp error:\n{e.stderr.decode()}")
        return None

# ----------------------------
# Build vector store for permanent PDFs (optional)
# ----------------------------
def build_custom_vectorstore():
    PDF_PATHS = []  # Add permanent PDFs here if needed
    docs = []
    for p in PDF_PATHS:
        loader = PyPDFLoader(p)
        docs.extend(loader.load())
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectors = embed_texts([c.page_content for c in chunks])
    return FAISS.from_embeddings(list(zip(chunks, vectors)))

custom_vs = build_custom_vectorstore()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📄 PDF, 🖼️ Image & 🎥 YouTube Q&A Bot")
uploads = st.file_uploader("Upload PDFs or Images", type=["pdf","png","jpg","jpeg","webp"], accept_multiple_files=True)
query = st.text_input("Ask a question or paste a YouTube link").strip()

external_docs = []
use_external = False

# Handle uploads
if uploads:
    use_external = True
    os.makedirs("uploaded_files", exist_ok=True)
    for f in uploads:
        path = os.path.join("uploaded_files", f.name)
        with open(path, "wb") as w:
            w.write(f.getbuffer())
        if f.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            external_docs.extend(loader.load())
        else:
            text = pytesseract.image_to_string(Image.open(path), config="--psm 6").strip()
            if text:
                external_docs.append(Document(page_content=text))

# Handle YouTube
yt_url, yt_id = extract_youtube_info(query)
if yt_id:
    use_external = True
    yt_text = fetch_youtube_transcript(yt_id, yt_url)
    if yt_text:
        external_docs.append(Document(page_content=yt_text))
        query = query.replace(yt_url, "").strip()

# ----------------------------
# Setup LLM
# ----------------------------
llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

# ----------------------------
# Answer/Summarize
# ----------------------------
if query:
    if use_external:
        if not external_docs:
            st.error("⚠️ No valid content found in uploaded files or YouTube.")
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(external_docs)
            vectors = embed_texts([c.page_content for c in chunks])
            vs = FAISS.from_embeddings(list(zip(chunks, vectors)))
            prompt = ChatPromptTemplate.from_template(
                "Answer based only on the context below.\n<context>{context}</context>\nQuestion: {input}"
            )
            doc_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vs.as_retriever()
            chain = create_retrieval_chain(retriever, doc_chain)
            resp = chain.invoke({"input": query})
            st.write(resp["answer"])
    else:
        if custom_vs:
            prompt = ChatPromptTemplate.from_template(
                "Answer based only on the context below.\n<context>{context}</context>\nQuestion: {input}"
            )
            doc_chain = create_stuff_documents_chain(llm, prompt)
            retriever = custom_vs.as_retriever()
            chain = create_retrieval_chain(retriever, doc_chain)
            resp = chain.invoke({"input": query})
            ans = resp.get("answer", "").strip()
            if ans.lower() != "i don't know" and ans:
                st.write(ans)
            else:
                st.write(llm.invoke(query).content)
        else:
            st.write(llm.invoke(query).content)
