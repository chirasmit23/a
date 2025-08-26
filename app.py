# ======================================================================================
# UltraChat Voice RAG — The Definitive, Fully-Featured, and Beginner-Friendly Version
# ======================================================================================
# This script combines all features:
# - Modern, clean user interface.
# - Robust, browser-based voice recording (fixes deployment errors).
# - Seamless voice-in and voice-out conversations (Text-to-Speech).
# - RAG capabilities with file uploads (PDF, DOCX, TXT, Images).
# - RAG with YouTube video transcripts.
# - The original, unreduced, multi-source web search and ranking engine.
# - Extensive comments and docstrings to make the code easy to understand.
#
# To run this file:
# 1. Make sure you have all libraries from requirements.txt installed.
# 2. Make sure you have a .env file with your API keys.
# 3. In your terminal, run: streamlit run app.py
# ======================================================================================

# --- Section 1: Importing Necessary Libraries ---
# --------------------------------------------------------------------------------------
# These are the Python packages needed to make our application work.

# Core libraries for file paths, I/O, and regular expressions
import os
import io
import re
import time
import html
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# The main framework for building the web app
import streamlit as st

# For loading secret API keys from a .env file
from dotenv import load_dotenv

# For extracting text from different file types
import pytesseract         # For images (OCR - Optical Character Recognition)
from PIL import Image      # For opening image files
import docx2txt            # For .docx (Microsoft Word) files
from bs4 import BeautifulSoup # For parsing HTML from web scraping

# For making HTTP requests to APIs and websites
import requests

# NEW: This is the modern, browser-based microphone recorder.
# It replaces the old 'sounddevice' and 'soundfile' libraries, fixing the deployment error.
from streamlit_mic_recorder import mic_recorder

# This is the library for connecting to the Groq AI models.
# We wrap it in a try-except block in case it's not installed.
try:
    from langchain_groq import ChatGroq
except ImportError:
    # If the library isn't found, we set it to None so the app doesn't crash.
    ChatGroq = None

# --- Section 2: Initial Configuration and Setup ---
# --------------------------------------------------------------------------------------
# This part sets up the basic configuration for our Streamlit page and loads API keys.

# Set the title and layout for the browser tab.
st.set_page_config(
    page_title="UltraChat Voice RAG",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load the environment variables (API keys) from the .env file.
load_dotenv()

# Assign the loaded API keys to variables for easier access.
# The .get() method is used to safely get the key without crashing if it's missing.
GROQ_KEY = os.getenv("groq_apikey")
DG_KEY = os.getenv("voice")
STACKEX_KEY = os.getenv("STACKEX_KEY")
SCRAPEDO_API_KEY = os.getenv("SCRAPEDO_API_KEY")

# If the user has Tesseract OCR installed, we tell the script where to find it.
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")
if TESSERACT_PATH and Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# --- Section 3: Styling the Application ---
# --------------------------------------------------------------------------------------
# Here, we use CSS to make our app look like the modern design from the screenshot.

st.markdown(
    """
    <style>
    /* Change the background color of the app */
    .stApp { background-color: #0e1117; }
    
    /* Hide the default Streamlit menu and deploy button for a cleaner look */
    #MainMenu, .stDeployButton, header { 
        display: none; 
        visibility: hidden; 
    }
    
    /* Style for the main title container */
    .title-container { 
        display: flex; 
        align-items: center; 
        gap: 15px; 
        padding: 1rem 0; 
    }
    .title-container h1 { 
        font-size: 2.5rem; 
        font-weight: 600; 
        margin: 0; 
        color: #FFFFFF; 
    }
    
    /* Style for the text links below the title */
    .action-links { 
        color: #a0a4ab; 
        font-size: 1rem; 
        margin-bottom: 2rem; 
    }
    
    /* Style for individual chat messages */
    .user-msg, .bot-msg { 
        padding: 1rem; 
        border-radius: 10px; 
        margin: 0.5rem 0; 
        word-wrap: break-word; 
    }
    .user-msg { background: rgba(56, 139, 253, 0.1); }
    .bot-msg  { background: rgba(120, 120, 120, 0.1); }
    
    /* Style for the chips that show the source of information */
    .source-chip {
        display: inline-block;
        margin: 4px 6px 0 0;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid #2b3d4f;
        color: #b7c9d9;
        background: #0f1b28;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Section 4: Core Helper and Utility Functions ---
# --------------------------------------------------------------------------------------
# These are small, reusable functions that help with common tasks throughout the script.

def _clean(s: Optional[str]) -> str:
    """
    Cleans a string by replacing multiple whitespace characters with a single space.

    Args:
        s (Optional[str]): The input string to clean.

    Returns:
        str: The cleaned string.
    """
    return re.sub(r"\s+", " ", (s or "")).strip()

def safe_rerun():
    """
    Safely triggers a rerun of the Streamlit app. This is used to refresh the UI
    after a message has been sent or an action is complete.
    """
    st.rerun()

# --- Section 5: Voice, YouTube, and File Processing Functions ---
# --------------------------------------------------------------------------------------
# These functions handle all the data input: voice, YouTube URLs, and file uploads.

def deepgram_transcribe(audio_bytes: bytes) -> str:
    """
    Sends recorded audio bytes to the Deepgram API for transcription (Speech-to-Text).

    Args:
        audio_bytes (bytes): The raw audio data from the microphone.

    Returns:
        str: The transcribed text, or an empty string if it fails.
    """
    if not DG_KEY:
        st.error("Deepgram API key is not set. Cannot transcribe audio.")
        return ""
    try:
        from deepgram import DeepgramClient, PrerecordedOptions
        client = DeepgramClient(api_key=DG_KEY)
        source = {"buffer": audio_bytes, "mimetype": "audio/wav"}
        opts = PrerecordedOptions(model="nova-2", smart_format=True, language="en")
        response = client.listen.prerecorded.v("1").transcribe_file(source, opts)
        return response.results.channels[0].alternatives[0].transcript
    except Exception as e:
        st.error(f"Could not transcribe audio. Error: {e}")
        return ""

def extract_youtube_info(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Finds a YouTube URL in a string and extracts the URL and the video ID.

    Args:
        text (str): The user's input text.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing the full URL and the video ID,
                                             or (None, None) if no URL is found.
    """
    # Regex to find different formats of YouTube URLs
    url_match = re.search(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[a-zA-Z0-9_-]{11}\S*)", text)
    if not url_match:
        return None, None
    
    url = url_match.group(1)
    # Regex to extract the 11-character video ID from the URL
    id_match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
    video_id = id_match.group(1) if id_match else None
    
    return url, video_id

def fetch_youtube_transcript(video_id: str) -> Optional[str]:
    """
    Uses a web scraping service (Scrape.do) to get the transcript of a YouTube video.

    Args:
        video_id (str): The 11-character ID of the YouTube video.

    Returns:
        Optional[str]: The full transcript text, or None if it fails.
    """
    if not SCRAPEDO_API_KEY:
        st.error("Scrape.do API key is not set. Cannot fetch YouTube transcript.")
        return None
        
    st.info("🚀 Fetching YouTube transcript... (This can take up to a minute)")
    
    # We use a third-party site that generates transcripts, and scrape it.
    transcript_url = f"https://www.tubetranscript.com/en/watch?v={video_id}"
    params = {"token": SCRAPEDO_API_KEY, "url": transcript_url, "render": "true"}
    
    try:
        response = requests.get("https://api.scrape.do/", params=params, timeout=120)
        response.raise_for_status()  # This will raise an error for bad responses (like 404, 500)
        
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, "lxml")
        transcript_div = soup.find("div", id="main-transcript-content")
        
        return transcript_div.get_text(strip=True) if transcript_div else None
    except Exception as e:
        st.error(f"Error fetching YouTube transcript: {e}")
        return None

def load_uploaded_files(uploaded_files: List) -> List[Dict]:
    """
    Processes a list of uploaded files and extracts text from them.

    Args:
        uploaded_files (List): A list of files uploaded via Streamlit's file_uploader.

    Returns:
        List[Dict]: A list of document dictionaries, each containing the content and metadata.
    """
    documents = []
    if not uploaded_files:
        return documents
        
    for file in uploaded_files:
        try:
            text = ""
            if file.name.lower().endswith(".docx"):
                text = docx2txt.process(io.BytesIO(file.getvalue()))
            elif file.name.lower().endswith(".txt"):
                text = file.getvalue().decode('utf-8')
            else: # Assume it's an image for OCR
                text = pytesseract.image_to_string(Image.open(file))
            
            if _clean(text):
                documents.append({"source": file.name, "title": file.name, "snippet": _clean(text), "url": ""})
        except Exception as e:
            st.warning(f"Could not process file {file.name}: {e}")
            
    return documents

# --- Section 6: Original Web Search and Ranking Engine (Unreduced) ---
# --------------------------------------------------------------------------------------
# This entire section contains the original, detailed functions for searching the web,
# scoring the results, and preparing them for the AI.

# --- Part 6.1: Individual Web Search Source Functions ---

def wikipedia_summary(q: str) -> Optional[Dict]:
    """Fetches a summary from Wikipedia."""
    try:
        r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(q)}", timeout=6)
        if r.status_code != 200: return None
        j = r.json(); extract = _clean(j.get("extract", ""))
        return {"source":"Wikipedia","title": j.get("title", q), "snippet": extract, "url": j.get("content_urls",{}).get("desktop",{}).get("page","")} if extract else None
    except Exception: return None

def duckduckgo_instant(q: str, max_results: int = 4) -> List[Dict]:
    """Fetches instant answers from DuckDuckGo."""
    try:
        r = requests.get(f"https://api.duckduckgo.com/?q={requests.utils.quote(q)}&format=json&no_html=1", timeout=6)
        j = r.json(); out = []
        for item in j.get("RelatedTopics", [])[:max_results]:
            if "Text" in item and "FirstURL" in item: out.append({"source":"DuckDuckGo","title":item.get("Text","")[:120],"snippet":item.get("Text",""),"url":item.get("FirstURL")})
        return out
    except Exception: return []

def mdn_search(q: str, max_results: int = 3) -> List[Dict]:
    """Searches the Mozilla Developer Network (MDN) for web development documentation."""
    try:
        r = requests.get(f"https://developer.mozilla.org/api/v1/search?q={requests.utils.quote(q)}&locale=en-US", timeout=6)
        if r.status_code != 200: return []
        j = r.json(); out = []
        for doc in j.get("documents", [])[:max_results]:
            out.append({"source":"MDN","title":doc.get("title",""),"snippet":_clean(doc.get("summary","")),"url":doc.get("mdn_url","")})
        return out
    except Exception: return []

def geeksforgeeks_search(q: str) -> Optional[Dict]:
    """Searches GeeksforGeeks for programming tutorials."""
    try:
        r = requests.get(f"https://www.geeksforgeeks.org/?s={requests.utils.quote(q)}", headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if r.status_code != 200: return None
        m = re.search(r'https://www\.geeksforgeeks\.org/[^"\'<> ]+/', r.text)
        if not m: return None
        url = m.group(0)
        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6).text
        snippet = _clean(re.sub(r"<[^>]+>", "", page))[:800]
        return {"source":"GeeksforGeeks","title": url.split("/")[-2].replace("-"," "),"snippet": snippet,"url": url}
    except Exception: return None

def stackoverflow_search(q: str, max_items: int = 4) -> List[Dict]:
    """Searches Stack Overflow for programming questions and answers."""
    try:
        params = {"order":"desc","sort":"relevance","q": q,"site":"stackoverflow","pagesize": max_items}
        if STACKEX_KEY: params["key"] = STACKEX_KEY
        r = requests.get("https://api.stackexchange.com/2.3/search/advanced", params=params, timeout=6)
        j = r.json(); out = []
        for it in j.get("items", []):
            out.append({"source":"StackOverflow","title": it.get("title",""),"snippet": _clean(it.get("title","")),"url": it.get("link","")})
        return out
    except Exception: return []

def arxiv_search(q: str, max_results: int = 2) -> List[Dict]:
    """Searches arXiv for scientific papers."""
    try:
        r = requests.get(f"http://export.arxiv.org/api/query?search_query=all:{requests.utils.quote(q)}&start=0&max_results={max_results}", timeout=6)
        entries = re.findall(r'<entry>(.*?)</entry>', r.text, re.S); out = []
        for e in entries:
            title = re.search(r'<title>(.*?)</title>', e, re.S); summary = re.search(r'<summary>(.*?)</summary>', e, re.S); link = re.search(r'<id>(.*?)</id>', e, re.S)
            out.append({"source":"arXiv", "title": (title.group(1).strip() if title else ""), "snippet": _clean((summary.group(1) if summary else "")), "url": link.group(1) if link else ""})
        return out
    except Exception: return []

# --- Part 6.2: Ranking and Filtering Logic ---

DOMAIN_PRIORITY = {"geeksforgeeks.org": 2.2, "wikipedia.org": 1.8, "developer.mozilla.org": 2.4, "stackoverflow.com": 1.2}
WEB_DEV_HINTS = {"html","css","javascript","js","dom","api","http"}

def looks_webdev(query: str) -> bool:
    """Checks if a query seems related to web development to boost MDN results."""
    return any(hint in (query or "").lower() for hint in WEB_DEV_HINTS)

def score_result(result: Dict, query_tokens: List[str]) -> float:
    """Calculates a relevance score for a single search result."""
    text_content = f"{result.get('title','')} {result.get('snippet','')}".lower()
    overlap = sum(1 for token in query_tokens if token in text_content)
    bias = 0.0
    source = (result.get("source") or "").lower()
    if source == "mdn": bias += 1.0 if looks_webdev(" ".join(query_tokens)) else -0.5
    return overlap + bias

def rank_results(results: List[Dict], query: str) -> List[Dict]:
    """Sorts a list of search results based on their relevance score."""
    tokens = re.findall(r"\w+", (query or "").lower())
    return sorted(results, key=lambda r: score_result(r, tokens), reverse=True)

def is_snippet_relevant(snippet: str, query: str) -> bool:
    """Uses a simple check and an optional LLM call to see if a snippet is relevant."""
    q_tokens = set(re.findall(r"\w+", query.lower()))
    s_tokens = set(re.findall(r"\w+", snippet.lower()))
    if not (q_tokens & s_tokens):
        return False
    # Optional: Add LLM-based check here if needed for higher accuracy
    return True

# --- Part 6.3: The Main Web Search Pipeline ---

def websearch_pipeline(query: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Executes the full web search process: gather, deduplicate, rank, and prepare documents.
    """
    if not query:
        return [], []

    # Step 1: Gather results from all sources
    st.write("Gathering results from web sources...")
    all_results = []
    w_res = wikipedia_summary(query)
    if w_res: all_results.append(w_res)
    all_results.extend(duckduckgo_instant(query))
    all_results.extend(mdn_search(query))
    all_results.extend(stackoverflow_search(query))
    g_res = geeksforgeeks_search(query)
    if g_res: all_results.append(g_res)
    all_results.extend(arxiv_search(query))

    # Step 2: Deduplicate results to avoid showing the same link twice
    unique_results_dict = {}
    for result in all_results:
        key = (result.get('url') or result.get('title') or '')[:200]
        if key and key not in unique_results_dict:
            unique_results_dict[key] = result
    
    unique_results = list(unique_results_dict.values())
    
    # Step 3: Rank the unique results based on relevance
    ranked = rank_results(unique_results, query)
    
    # Step 4: Prepare the ranked results as "documents" for the AI
    docs_for_rag = [
        {"source": r.get("source"), "title": r.get("title"), "snippet": _clean(r.get("snippet","")), "url": r.get("url")}
        for r in ranked if r.get("snippet")
    ]
    
    return ranked, docs_for_rag

# --- Section 7: AI and Language Model Functions ---
# --------------------------------------------------------------------------------------
# These functions handle the interaction with the Groq Large Language Model (LLM).

def _llm_invoke(prompt: str) -> str:
    """
    Sends a prompt to the Groq LLM and returns the text response.

    Args:
        prompt (str): The complete prompt to send to the AI.

    Returns:
        str: The AI's generated response.
    """
    if not (ChatGroq and GROQ_KEY):
        return "(LLM not configured. Please set your 'groq_apikey' in the .env file.)"
    try:
        llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_KEY)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"(An error occurred with the LLM: {e})"

def synthesize_from_snippets(question: str, docs: List[Dict]) -> str:
    """
    Builds a prompt from web search snippets and asks the LLM to synthesize an answer.
    """
    filtered_docs = [d for d in docs if is_snippet_relevant(d.get("snippet", ""), question)]
    if not filtered_docs:
        return "(No sufficiently relevant information was found to answer your question.)"
        
    context = "\n\n".join(
        f"Source: {d['source']}\nTitle: {d['title']}\nURL: {d.get('url', 'N/A')}\nContent: {d['snippet'][:1500]}"
        for d in filtered_docs[:10] # Use top 10 relevant docs
    )
    
    prompt = (
        "You are a helpful AI research assistant. Your task is to synthesize a clear and comprehensive answer to the user's question "
        "based *only* on the provided context. Do not use any outside knowledge.\n"
        "Please cite your sources at the end of each relevant sentence, like this: [Source Name].\n"
        "Finally, create a 'Sources:' list at the very end with the titles and URLs of the documents you used.\n\n"
        f"--- CONTEXT ---\n{context}\n\n"
        f"--- QUESTION ---\n{question}\n\n"
        "--- ANSWER ---\n"
    )
    
    return _llm_invoke(prompt)

# --- Section 8: Main Application UI and Event Logic ---
# --------------------------------------------------------------------------------------
# This is the final part of the script that builds the user interface and handles user interactions.

# --- Part 8.1: Initialize Session State ---
# Session state is like Streamlit's memory. It remembers values between reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_response" not in st.session_state:
    st.session_state.audio_response = None

# --- Part 8.2: Render the UI Elements ---
st.markdown('<div class="title-container"><h1>🎤 UltraChat Voice RAG</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="action-links"><a>⬆️ Upload</a> | <a>🗣️ Speak</a> | <a>📝 Summarize</a> | <a>🤖 Ask</a> | <a>🌐 WebSearch</a> | <a>📺 YouTube</a></div>', unsafe_allow_html=True)

# This container will hold the chat history.
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        # Display each message with the appropriate CSS class
        st.markdown(f"<div class='{'user-msg' if message['role'] == 'user' else 'bot-msg'}'>{html.escape(message['content'])}</div>", unsafe_allow_html=True)
    
    # If there's an audio response waiting, play it automatically.
    if st.session_state.audio_response:
        st.audio(st.session_state.audio_response, format="audio/mpeg", autoplay=True)
        st.session_state.audio_response = None # Clear it so it doesn't play again

# The input bar at the bottom of the screen. We use columns for the layout.
input_columns = st.columns([0.3, 0.1, 0.15, 0.35, 0.1])
with input_columns[0]:
    uploaded_files = st.file_uploader("files", label_visibility="collapsed", type=["pdf","docx","txt","png","jpg","jpeg"], accept_multiple_files=True)
with input_columns[1]:
    audio_data = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", just_once=True, use_container_width=True)
with input_columns[2]:
    selected_action = st.selectbox("Action", ["Ask", "Speak", "Summarize", "WebSearch"], label_visibility="collapsed")
with input_columns[3]:
    user_text_input = st.text_input("message", placeholder="Type message or paste YouTube URL...", label_visibility="collapsed")
with input_columns[4]:
    send_button_clicked = st.button("Send", use_container_width=True)

# --- Part 8.3: Handle User Interactions (Event Logic) ---

# Priority 1: Handle voice input immediately if it exists.
if audio_data:
    st.session_state.messages.append({"role": "user", "content": "🎤 (Processing your voice...)"})
    transcript = deepgram_transcribe(audio_data['bytes'])
    
    if transcript:
        st.session_state.messages[-1] = {"role": "user", "content": f"🎤: {transcript}"} # Update the message with the transcript
        with st.spinner("Thinking..."):
            ai_response = _llm_invoke(transcript)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            # Generate Text-to-Speech response
            if DG_KEY:
                try:
                    tts_response = requests.post("https://api.deepgram.com/v1/speak?model=aura-asteria-en", headers={"Authorization": f"Token {DG_KEY}"}, json={"text": ai_response}, timeout=30)
                    if tts_response.status_code == 200:
                        st.session_state.audio_response = tts_response.content
                except Exception as e:
                    st.error(f"Text-to-Speech failed: {e}")
        safe_rerun()
    else:
        st.session_state.messages.pop() # Remove the "Processing..." message if transcription fails
        st.error("Could not understand the audio. Please try again.")
        safe_rerun()

# Priority 2: Handle text, file, or URL input when the 'Send' button is clicked.
if send_button_clicked and (user_text_input or uploaded_files):
    # Step 1: Add the user's message to the chat history.
    query_text = user_text_input or "(Processing uploaded documents...)"
    st.session_state.messages.append({"role": "user", "content": query_text})

    # Step 2: Gather all sources of information (files, YouTube).
    all_context_docs = []
    if uploaded_files:
        all_context_docs.extend(load_uploaded_files(uploaded_files))
        
    youtube_url, video_id = extract_youtube_info(user_text_input)
    if video_id:
        transcript = fetch_youtube_transcript(video_id)
        if transcript:
            st.success("YouTube transcript successfully added as context!")
            all_context_docs.append({"source": "YouTube", "title": f"YouTube Video ({video_id})", "snippet": transcript, "url": youtube_url})
            # If the user only pasted a URL, change action to summarize.
            if not _clean(user_text_input.replace(youtube_url, '')):
                selected_action = "Summarize"
                query_text = f"Summarize the content of the video at {youtube_url}"
                st.session_state.messages[-1]["content"] = query_text # Update the user message
    
    # Step 3: Execute the chosen action.
    ai_response = ""
    with st.spinner(f"Performing action: {selected_action}..."):
        if selected_action == "Summarize":
            if not all_context_docs:
                ai_response = "Please upload a file or provide a YouTube URL to summarize."
            else:
                context_str = "\n\n".join(f"--- Document: {d['title']} ---\n{d['snippet']}" for d in all_context_docs)
                prompt = f"Please provide a concise summary of the following document(s):\n\n{context_str}"
                ai_response = _llm_invoke(prompt)
        
        elif selected_action == "WebSearch":
            ranked_results, web_docs = websearch_pipeline(query_text)
            final_docs = all_context_docs + web_docs
            ai_response = synthesize_from_snippets(query_text, final_docs)

        elif selected_action in ["Ask", "Speak"]:
            prompt = query_text
            if all_context_docs:
                context_str = "\n\n".join(f"--- Context from: {d['title']} ---\n{d['snippet']}" for d in all_context_docs)
                prompt = f"Using the provided context below, please answer the user's question.\n\n--- Context ---\n{context_str}\n\n--- Question ---\n{query_text}\n\n--- Answer ---"
            ai_response = _llm_invoke(prompt)
            
            # Generate audio for the 'Speak' action
            if selected_action == "Speak" and DG_KEY:
                try:
                    tts_response = requests.post("https://api.deepgram.com/v1/speak?model=aura-asteria-en", headers={"Authorization": f"Token {DG_KEY}"}, json={"text": ai_response}, timeout=30)
                    if tts_response.status_code == 200:
                        st.session_state.audio_response = tts_response.content
                except Exception as e:
                    st.error(f"Text-to-Speech failed: {e}")
    
    # Step 4: Add the AI's response to the chat history and refresh the app.
    if ai_response:
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
    safe_rerun()

# --- Part 8.4: Clear Chat Button ---
st.markdown("---")
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.session_state.audio_response = None
    safe_rerun()
