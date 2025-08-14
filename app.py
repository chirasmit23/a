import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
import pytesseract
from PIL import Image
import cv2
from youtube_transcript_api import YouTubeTranscriptApi

# Load API key
load_dotenv()
groq_api_key = os.getenv("groq_apikey")

# Helper function: Extract YouTube video ID
def extract_youtube_id(url):
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# Helper function: Fetch transcript text
def fetch_youtube_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        try:
            
            transcript = api.fetch(video_id,languages=['en'])  # Old API style
            return " ".join(entry.text for entry in transcript)
        except:
            transcript = api.fetch(video_id,languages=['hi'])  # Old API style
            return " ".join(entry.text for entry in transcript)
    except Exception as e:
        return f"Error fetching transcript: {e}"

# UI
st.title("📄 PDF, 🖼️ Image & 🎥 YouTube Question Answer Bot")
uploaded_files = st.file_uploader(
    "Upload PDFs or Images", 
    type=["pdf", "png", "jpg", "jpeg", "webp"], 
    accept_multiple_files=True
)
query = st.text_input("Ask a question about your files or paste a YouTube link")

if uploaded_files or query:
    os.makedirs("uploaded_files", exist_ok=True)
    all_docs = []

    # Handle uploaded PDFs & Images
    for uploaded_file in uploaded_files:
        file_path = os.path.join("uploaded_files", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File saved: {uploaded_file.name}")

        if uploaded_file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
        else:
            extracted_text = pytesseract.image_to_string(Image.open(file_path), config="--psm 6")
            if extracted_text.strip():
                all_docs.append(Document(page_content=extracted_text))
            else:
                st.warning(f"No text found in {uploaded_file.name}")

    # Handle YouTube transcript if URL detected
    youtube_id = extract_youtube_id(query)
    if youtube_id:
        
        yt_text = fetch_youtube_transcript(youtube_id)
        if yt_text.startswith("Error"):
            st.error(yt_text)
        else:
            all_docs.append(Document(page_content=yt_text))
            

    # Proceed if we have documents or transcript
    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(all_docs)

        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        vectorstore = FAISS.from_documents(final_documents, embeddings)

        llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the provided context. 
            If no question is given, provide a concise summary instead.
            <context>{context}</context>
            Question: {input}"""
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectorstore.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        if query:
            result = retrieval_chain.invoke({"input": query})
        else:
            result = retrieval_chain.invoke({"input": "Summarize the content."})

        st.subheader("Answer / Summary:")
        st.write(result["answer"])
