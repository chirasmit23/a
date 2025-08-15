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
# --- NEW: Import the dedicated summarization chain ---
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
nomic_api_key = os.getenv("nomic_api")
scrapedo_api_key = os.getenv("SCRAPEDO_API_KEY")

if not all([groq_api_key, nomic_api_key, scrapedo_api_key]):
    st.error("API keys for Groq, Nomic, and Scrape.do must be set in your environment secrets.")
    st.stop()

llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
embeddings_model = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key=nomic_api_key)

# ----------------------------
# All your helper functions (extract_youtube_info, fetch_transcript_with_remote_actions)
# remain unchanged. They are working perfectly. I will omit them here for brevity,
# but they should remain in your final file.
def extract_youtube_info(text):
    # ... (your existing function)
def fetch_transcript_with_remote_actions(youtube_url: str) -> str | None:
    # ... (your existing, working function)
# ----------------------------

# ... (Paste your extract_youtube_info and fetch_transcript_with_remote_actions functions here) ...

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
    # --- Store the original, full query before we modify it ---
    original_query = query 
    
    if yt_url and yt_id:
        with st.spinner("Attempting to fetch YouTube transcript using remote browser automation..."):
            yt_text = fetch_transcript_with_remote_actions(yt_url)
            if yt_text:
                external_docs.append(Document(page_content=yt_text, metadata={"source": "YouTube"}))
                # Clean the query of the URL for intent detection
                query = query.replace(yt_url, "").strip()

    if uploads:
        # ... (your file upload logic remains the same) ...

    # --- REWRITTEN: Smart Intent-Based Processing ---
    if external_docs:
        with st.spinner("Analyzing content and preparing answer..."):
            # 1. Define summarization keywords
            summarization_keywords = ["summarize", "summary", "overview", "key points", "tldr", "main ideas"]
            # Check if the query is a summarization request, or if it was empty after removing the URL
            is_summary_request = any(keyword in query.lower() for keyword in summarization_keywords) or not query

            # 2. Split documents into chunks (needed for both paths)
            splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
            chunks = splitter.split_documents(external_docs)

            if is_summary_request:
                # --- Path A: Summarization ---
                st.info("Summarization request detected. Generating summary...")
                
                # Use the dedicated summarization chain
                summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
                summary = summary_chain.run(chunks)
                st.write(summary)

            else:
                # --- Path B: Specific Q&A (Your original RAG logic) ---
                st.info("Specific question detected. Searching for the answer...")
                
                vs = Chroma.from_documents(chunks, embeddings_model)
                retriever = vs.as_retriever()
                
                prompt = ChatPromptTemplate.from_template("Answer the user's question based only on the provided context.\n\n<context>{context}</context>\n\nQuestion: {input}")
                doc_chain = create_stuff_documents_chain(llm, prompt)
                chain = create_retrieval_chain(retriever, doc_chain)
                
                resp = chain.invoke({"input": query})
                st.write(resp["answer"])
    
    elif not yt_id: # Handle general queries if no files or links are provided
        st.write("Answering general question...")
        st.write(llm.invoke(original_query).content)
