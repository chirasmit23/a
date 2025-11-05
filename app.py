import os,re,requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from google.genai import types

# ✅ Loaders, Splitters, Embeddings, VectorStores
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
# ✅ Prompt Templates
from langchain_core.prompts import ChatPromptTemplate

# ✅ Chains (these are now inside langchain.chains)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# ✅ Chat Models (external providers)
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ OCR / Image handling
# import pytesseract
# from PIL import Image
# import cv2
from google import genai
from googleapiclient.discovery import build

# Load API keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
SCRAPEDO_API_KEY = os.getenv("SCRAPEDO_API_KEY")
groq_api_key = os.getenv("groq_apikey")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# UI
st.title("PDF & Image Question Answer Bot")
choose=st.selectbox("option",["select","Websearch","summarisation"])
def _clean(text: str) -> str:
    """A helper function to clean up scraped text by removing excess whitespace."""
    return re.sub(r"\s+", " ", text).strip()
def extract_youtube_id(url):
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None
def extract_youtube_url(url):
    match= re.search(r"(https?://(?:www\.)?(?:youtube\.com|youtu\.be)/(?:watch\?v=|shorts/|embed/)?[a-zA-Z0-9_-]{11}(?:[^\s]*)?)",url)

    return match.group(1) if match else None

# Helper function: Fetch transcript text
def fetch_youtube_transcript(video_id):
    if not SCRAPEDO_API_KEY:
        return None
    try:
        url = f"https://www.tubetranscript.com/en/watch?v={video_id}"
        params = {"token": SCRAPEDO_API_KEY, "url": url, "render": "true"}
        r = requests.get("https://api.scrape.do/", params=params, timeout=60)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        div = soup.find("div", id="main-transcript-content")
        return _clean(div.get_text(" ", strip=True)) if div else None
    
    except Exception as e:
        return f"Error fetching transcript: {e}"

def google_custom_search(query):
    """Performs a Google search and returns formatted results with sources."""
    if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
        return "Error: Google Custom Search API key or Search Engine ID not configured.", []
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=5).execute()
        items = res.get("items", [])
        
        if not items:
            return "No search results found.", []    
        formatted_results = ""
        sources = []
        for i, item in enumerate(items):
            title = item.get("title", "No title")
            snippet = item.get("snippet", "No description")
            link = item.get("link", "")
            formatted_results += f"Result {i+1}:\nTitle: {title}\nDescription: {snippet}\nURL: {link}\n\n"
            sources.append(link)
        
        return formatted_results, sources
    except Exception as e:
        return f"Error calling Custom Search API: {e}", []
uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg", "webp"])
query = st.text_input("Ask a question about your document or image")

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

if uploaded_file:
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved: {uploaded_file.name}")

    docs = []

    if uploaded_file.name.lower().endswith(".pdf"):
        # Load PDF
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    else:
        # Load Image and extract text
        try:
            client = genai.Client(api_key=gemini_api_key)
            # extracted_text = pytesseract.image_to_string(Image.open(file_path), config="--psm 6")
            # if extracted_text and extracted_text.strip():
            #     docs = [Document(page_content=extracted_text)]
            # else:
            #     st.error("No text found in the image.")
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            my_file = client.files.upload(file=file_path)

            # Generate content
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[my_file, query],
            )
            if response.text:
                docs = [Document(page_content=response.text)]
                
                st.write(docs[0].page_content)
        except Exception as e:
            st.error(f"Error processing image: {e}")


    
    if docs and uploaded_file.name.lower().endswith(".pdf") :
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        # Create embeddings & vectorstore
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = FAISS.from_documents(final_documents, embeddings)

        # Create LLM & retrieval chain
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key= gemini_api_key)
        prompt = ChatPromptTemplate.from_template(
            """Answer the following question based only on the provided context. 
            Think step by step before providing a detailed answer.
            <context>{context}</context>
            Question: {input}"""
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectorstore.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

elif choose=="Websearch" and query:
    search_results, sources = google_custom_search(query)
    st.info("Found sources. Generating answer...")
    search_llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key)
    search_prompt_template = """
    Based on the following search results, provide a comprehensive answer to the user's question.
    After the answer, list the URLs of the sources you used under a 'Sources:' heading.

    Search Results:
    {context}

    User's Question: {question}
    """
    search_prompt = ChatPromptTemplate.from_template(search_prompt_template)
    search_chain = search_prompt | search_llm
    response = search_chain.invoke({"context": search_results, "question": query})
    st.subheader("Answer from Web Search:")
    st.write(response.content)

elif choose=="summarisation" and query: 
    try:       
        st.info("fetch via gemini")
        youtube_url=extract_youtube_url(query)
        if youtube_url:
            from google.genai.types import HttpOptions, Part
            
            client = genai.Client(http_options=HttpOptions(api_version="v1beta"),api_key=gemini_api_key)
            model_id = "gemini-2.5-flash"

            response = client.models.generate_content(
                model=model_id,
                contents=[
                    Part.from_uri(
                        file_uri=youtube_url,
                        mime_type="video/mp4",
                    ),
                    "Write a short and engaging blog post based on this video.",
                ],
            )

            st.write(response.text)
    except:        
        st.info("fetch via scrap")
            
        youtube_id = extract_youtube_id(query)
        if youtube_id:
            st.info("fetching youtube details")
            
            yt_text = fetch_youtube_transcript(youtube_id)
            if:
                docs = [Document(page_content=yt_text)]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(docs)
                embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                vectorstore = FAISS.from_documents(final_documents, embeddings)
                llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key)
                prompt = ChatPromptTemplate.from_template(
                    "You are a YouTube video summarizer. Answer the user's question based on the video transcript provided.\n<context>{context}</context>\nQuestion: {input}"
                )
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = vectorstore.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                result = retrieval_chain.invoke({"input": query})
                st.subheader("Answer from YouTube Transcript:")
                st.write(result.get("answer", "No answer could be generated."))
                with st.expander("Show Transcript"):
                    st.write(yt_text)
            
           
else:
    if query:
        classifier_query = """
You are a query classifier.

Your task is to decide whether the user's question requires real-time information or can be answered using general knowledge.and give the answer of the query

Label the question as one of the following:
- 'realtime' → The question depends on current or time-sensitive data (e.g., "Who is the current president of the USA?", "What is the latest news in AI?", "Today’s weather in Delhi").
- 'general' → The question can be answered from general, timeless knowledge (e.g., "What is photosynthesis?", "Who wrote Hamlet?", "Explain quantum entanglement").

 Respond with only one word: 'realtime' or 'general"
"""
        classifier_llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key)
        classifier_promt=ChatPromptTemplate.from_messages([("system", classifier_query), ("human", "{question}")])
        classifier_chain=classifier_promt|classifier_llm
        classification_result = classifier_chain.invoke(query)
        query_type = classification_result.content.strip().lower()
        
        

    
    
        if "realtime" in query_type:
            grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

            config = types.GenerateContentConfig(
                tools=[grounding_tool]
            )
            client = genai.Client(api_key=gemini_api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[query],
                config=config,
            )

            
            st.write("**Answer:**", response.text)
            
        else:   
            st.info("search via chat")
            llm = ChatGroq(
                model="openai/gpt-oss-120b",
                api_key=groq_api_key
            )

            # Create prompt template
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user's question clearly and concisely.\nQuestion: {question}"
            )
            chain = prompt | llm
            response = chain.invoke({"question": query})
            st.subheader("Answer:")
            st.write(response.content)

            
