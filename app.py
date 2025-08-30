# app.py
# ======================================================================================
# UltraChat Voice RAG — Single-file Streamlit app with MCP-first WebSearch + fallbacks
# ======================================================================================

import os
import io
import re
import time
import json
import html
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import docx2txt
from bs4 import BeautifulSoup
from fpdf import FPDF

# Browser mic recorder (preferred for deployed Streamlit)
# pip install streamlit-mic-recorder
from streamlit_mic_recorder import mic_recorder

# LangChain / Groq LLM (optional — wrapped)
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

# LangChain helpers for RAG (optional, used if you have them)
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.prompts import ChatPromptTemplate
    from langchain.docstore.document import Document
    from langchain_community.vectorstores import Chroma
    from langchain_nomic import NomicEmbeddings
except Exception:
    PyPDFLoader = None
    RecursiveCharacterTextSplitter = None
    create_stuff_documents_chain = None
    ChatPromptTemplate = None
    Document = dict
    Chroma = None
    NomicEmbeddings = None

# -----------------------------------------------------------------------------
# Config / Env
# -----------------------------------------------------------------------------
st.set_page_config(page_title="UltraChat Voice RAG", layout="wide")
load_dotenv()

GROQ_KEY = os.getenv("groq_apikey")             # Groq LLM key
DG_KEY = os.getenv("voice")                     # Deepgram (optional for TTS/STT)
SCRAPEDO_API_KEY = os.getenv("SCRAPEDO_API_KEY")
STACKEX_KEY = os.getenv("STACKEX_KEY")
MCP_URL = os.getenv("MCP_URL")  # Optional: MCP server base URL (e.g. http://localhost:3000 or https://mcp.example.com)
# If you want to run a free MCP web-search, see projects like:
# https://github.com/gabrimatic/mcp-web-search-tool or https://github.com/pskill9/web-search
# They expose a simple search endpoint and implement the MCP protocol.

# Tesseract OCR path (optional)
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")
if TESSERACT_PATH and Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# -----------------------------------------------------------------------------
# Styling (single title + single uploader)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background-color: #0e1117; color: #e6edf3; }
      #MainMenu, header { display: none; }
      .title { font-size: 2.4rem; font-weight:700; margin-bottom:0.2rem; color: #fff; }
      .subtitle { color: #a0a4ab; margin-bottom:1rem; }
      .user-msg, .bot-msg { padding: 12px; border-radius: 10px; margin: 8px 0; word-wrap:break-word; }
      .user-msg { background: rgba(56,139,253,0.08); }
      .bot-msg { background: rgba(200,200,200,0.04); }
      .source-chip { display:inline-block; margin:4px 6px 0 0; padding:4px 8px; border-radius:999px; border:1px solid #2b3d4f; color:#b7c9d9; background:#0f1b28; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🎤 UltraChat Voice RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload files for RAG, or choose WebSearch (MCP first) to fetch live facts. </div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def export_text_to_pdf_bytes(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, margin=10)
    pdf.set_font("Arial", size=12)
    safe = _clean(text)
    for line in safe.split("\n"):
        pdf.multi_cell(0, 8, line)
    return pdf.output(dest="S").encode("latin-1", "replace")

# -----------------------------------------------------------------------------
# Audio helpers (Deepgram STT / TTS optional)
# -----------------------------------------------------------------------------
def deepgram_transcribe(audio_bytes: bytes) -> str:
    if not DG_KEY:
        return ""
    try:
        from deepgram import DeepgramClient, PrerecordedOptions, FileSource
        client = DeepgramClient(api_key=DG_KEY)
        src = {"buffer": audio_bytes}
        opts = PrerecordedOptions(model="nova-2", smart_format=True, language="en")
        resp = client.listen.prerecorded.v("1").transcribe_file(src, opts)
        return resp.results.channels[0].alternatives[0].transcript if resp.results.channels else ""
    except Exception:
        return ""

def deepgram_tts(text: str) -> Optional[bytes]:
    if not DG_KEY:
        return None
    try:
        url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
        r = requests.post(url, headers={"Authorization": f"Token {DG_KEY}"}, json={"text": text}, timeout=30)
        if r.status_code == 200:
            return r.content
    except Exception:
        return None

# -----------------------------------------------------------------------------
# File / YouTube helpers
# -----------------------------------------------------------------------------
def load_uploaded_files(uploaded_files) -> List[Dict]:
    docs = []
    if not uploaded_files:
        return docs
    updir = Path("uploads"); updir.mkdir(exist_ok=True)
    for f in uploaded_files:
        p = updir / f.name
        p.write_bytes(f.getbuffer())
        text = ""
        try:
            if f.name.lower().endswith(".pdf") and PyPDFLoader:
                docs_from_pdf = PyPDFLoader(str(p)).load()
                for d in docs_from_pdf:
                    docs.append({"source": f.name, "title": f.name, "snippet": _clean(d.page_content[:4000]), "url": ""})
                continue
            elif f.name.lower().endswith(".docx"):
                text = docx2txt.process(str(p))
            elif f.name.lower().endswith(".txt"):
                text = p.read_text(errors="ignore")
            else:
                text = pytesseract.image_to_string(Image.open(p))
        except Exception:
            text = ""
        if _clean(text):
            docs.append({"source": f.name, "title": f.name, "snippet": _clean(text[:4000]), "url": ""})
    return docs

def extract_youtube_id(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", text)
    return m.group(1) if m else None

def fetch_youtube_transcript(video_id: str) -> Optional[str]:
    # Use scrape.do if SCRAPEDO_API_KEY provided; else return None
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
    except Exception:
        return None

# -----------------------------------------------------------------------------
# MCP client: try calling MCP server for websearch (if configured)
# -----------------------------------------------------------------------------
def mcp_search(query: str) -> List[Dict]:
    """
    Call an MCP-compatible websearch server if MCP_URL is set.
    This function tries a couple of reasonable request formats:
      1) POST MCP_URL/jsonrpc (JSON-RPC)
      2) POST MCP_URL/search with {"query":...}
      3) POST MCP_URL with {"query":...}
    The MCP server should return a list of results that we convert to dicts:
      {"source":..., "title":..., "snippet":..., "url":...}
    """
    if not MCP_URL:
        return []
    headers = {"Content-Type": "application/json"}
    payloads = [
        {"jsonrpc": "2.0", "method": "search", "params": {"query": query}, "id": 1},
        {"query": query},
        {"q": query},
    ]
    candidates = [
        MCP_URL,
        MCP_URL.rstrip("/") + "/search",
        MCP_URL.rstrip("/") + "/rpc",
        MCP_URL.rstrip("/") + "/invoke",
    ]
    for url in candidates:
        for payload in payloads:
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=10)
                if r.status_code != 200:
                    continue
                j = r.json()
                # Flexible parsing: check for several possible shapes
                results = []
                if isinstance(j, dict) and "result" in j and isinstance(j["result"], dict) and "results" in j["result"]:
                    raw = j["result"]["results"]
                elif isinstance(j, dict) and "results" in j:
                    raw = j["results"]
                elif isinstance(j, list):
                    raw = j
                elif isinstance(j, dict) and "data" in j:
                    raw = j["data"]
                else:
                    raw = None

                if not raw:
                    # if no structured result, try to parse "text" that contains HTML results
                    text = r.text
                    raw = None
                    # fallthrough: skip to next payload
                    continue

                for item in raw:
                    # item might be dict with 'title','snippet','url','source'
                    if not isinstance(item, dict):
                        continue
                    results.append({
                        "source": item.get("source") or item.get("engine") or "MCP",
                        "title": item.get("title") or item.get("name") or item.get("headline") or "",
                        "snippet": _clean(item.get("snippet") or item.get("summary") or item.get("text") or ""),
                        "url": item.get("url") or item.get("link") or item.get("first_url") or ""
                    })
                if results:
                    return results
            except Exception:
                continue
    return []

# -----------------------------------------------------------------------------
# Web search fallbacks (Wikipedia, DuckDuckGo HTML search, MDN, StackExchange, GeeksforGeeks)
# -----------------------------------------------------------------------------
def wikipedia_summary(q: str) -> Optional[Dict]:
    try:
        r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(q)}", timeout=6)
        if r.status_code != 200:
            return None
        j = r.json()
        extract = _clean(j.get("extract", ""))
        if not extract:
            return None
        return {"source": "Wikipedia", "title": j.get("title", q), "snippet": extract, "url": j.get("content_urls", {}).get("desktop", {}).get("page", "")}
    except Exception:
        return None

def duckduckgo_html_search(q: str, max_results: int = 4) -> List[Dict]:
    """
    Uses DuckDuckGo HTML endpoint to get text results (no JS).
    This endpoint is accessible without an API key.
    """
    try:
        url = "https://html.duckduckgo.com/html/"
        params = {"q": q}
        r = requests.post(url, data=params, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "lxml")
        results = []
        for a in soup.select("a.result__a")[:max_results]:
            href = a.get("href")
            title = _clean(a.get_text())
            # DuckDuckGo returns redirect URLs like /l/?kh=-1&uddg=<encoded-url>; try to parse
            link = href
            # Try to decode uddg param
            m = re.search(r"uddg=(https?%3A%2F%2F[^&]+)", href)
            if m:
                link = requests.utils.unquote(m.group(1))
            # Try to fetch snippet from surrounding element
            snippet_el = a.find_parent().select_one(".result__snippet")
            snippet = _clean(snippet_el.get_text()) if snippet_el else title
            results.append({"source": "DuckDuckGo", "title": title, "snippet": snippet, "url": link})
        return results
    except Exception:
        return []

def geeksforgeeks_site_search(q: str) -> Optional[Dict]:
    """
    Do a quick site:geeksforgeeks.org search via DuckDuckGo HTML, then fetch the article and extract simple text.
    This returns the first GfG article's snippet/full-text.
    """
    try:
        query = f"site:geeksforgeeks.org {q}"
        hits = duckduckgo_html_search(query, max_results=6)
        for h in hits:
            if "geeksforgeeks.org" in (h.get("url") or ""):
                # fetch article and extract main text (best-effort)
                try:
                    r = requests.get(h["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
                    soup = BeautifulSoup(r.text, "lxml")
                    # try to find article content blocks
                    article = soup.select_one("article") or soup.select_one(".entry-content") or soup.body
                    text = _clean(article.get_text(" ", strip=True))[:4000]
                    if text:
                        return {"source": "GeeksforGeeks", "title": h.get("title"), "snippet": text, "url": h.get("url")}
                except Exception:
                    continue
        return None
    except Exception:
        return None

def mdn_search(q: str, max_results: int = 3) -> List[Dict]:
    try:
        r = requests.get(f"https://developer.mozilla.org/api/v1/search?q={requests.utils.quote(q)}&locale=en-US", timeout=6)
        if r.status_code != 200:
            return []
        j = r.json()
        out = []
        for doc in j.get("documents", [])[:max_results]:
            out.append({"source": "MDN", "title": doc.get("title", ""), "snippet": _clean(doc.get("summary", "") or ""), "url": doc.get("mdn_url", "")})
        return out
    except Exception:
        return []

def stackoverflow_search(q: str, max_items: int = 4) -> List[Dict]:
    try:
        params = {"order": "desc", "sort": "relevance", "q": q, "site": "stackoverflow", "pagesize": max_items}
        if STACKEX_KEY:
            params["key"] = STACKEX_KEY
        r = requests.get("https://api.stackexchange.com/2.3/search/advanced", params=params, timeout=6)
        j = r.json()
        out = []
        for it in j.get("items", []):
            out.append({"source": "StackOverflow", "title": it.get("title", ""), "snippet": _clean(it.get("title", "")), "url": it.get("link", "")})
        return out
    except Exception:
        return []

# -----------------------------------------------------------------------------
# Ranking / dedupe / prepare docs
# -----------------------------------------------------------------------------
def rank_results(results: List[Dict], query: str) -> List[Dict]:
    tokens = re.findall(r"\w+", (query or "").lower())
    def score(r):
        txt = (r.get("title","") + " " + r.get("snippet","")).lower()
        overlap = sum(1 for t in tokens if t in txt)
        # small bias for authoritative sources
        bias = 0.0
        src = (r.get("source") or "").lower()
        if "mdn" in src: bias += 1.0
        if "wikipedia" in src: bias += 0.7
        if "geeksforgeeks" in src: bias += 0.9
        return overlap + bias
    return sorted(results, key=score, reverse=True)

def dedupe_results(results: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in results:
        key = (r.get("url") or r.get("title") or "")[:240]
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out

# -----------------------------------------------------------------------------
# Main websearch pipeline: MCP-first, then fallbacks
# -----------------------------------------------------------------------------
def websearch_pipeline(query: str) -> Tuple[List[Dict], List[Dict]]:
    """Return (ranked_results, docs_for_rag)"""
    if not query:
        return [], []
    # 1) MCP first (if configured)
    mcp_results = mcp_search(query) if MCP_URL else []
    results = []
    if mcp_results:
        results.extend(mcp_results)

    # 2) If MCP returned nothing, use open sources
    if not results:
        # Wikipedia
        w = wikipedia_summary(query)
        if w: results.append(w)
        # DuckDuckGo quick hits
        results.extend(duckduckgo_html_search(query, max_results=4))
        # MDN if webdev
        results.extend(mdn_search(query))
        # StackOverflow
        results.extend(stackoverflow_search(query))
        # GfG deep article search
        gfg = geeksforgeeks_site_search(query)
        if gfg: results.append(gfg)

    # 3) Deduplicate + rank
    results = dedupe_results(results)
    ranked = rank_results(results, query)

    # 4) Build docs for RAG / LLM
    docs = []
    for r in ranked:
        if r.get("snippet"):
            docs.append({"source": r.get("source"), "title": r.get("title"), "snippet": r.get("snippet"), "url": r.get("url")})
    return ranked, docs

# -----------------------------------------------------------------------------
# LLM invocation / synthesizer
# -----------------------------------------------------------------------------
def llm_invoke(prompt: str) -> str:
    if not (ChatGroq and GROQ_KEY):
        # fallback: return prompt-based short reply (non-LLM) so UI doesn't break
        return "(LLM not configured: set groq_apikey and install langchain_groq) " + prompt[:400]
    try:
        client = ChatGroq(groq_api_key=GROQ_KEY, model_name="qwen/qwen3-32b") if "qwen" in dir(ChatGroq) else ChatGroq(model="llama3-8b-8192", api_key=GROQ_KEY)
        # The ChatGroq interface varies — try common invocation patterns:
        try:
            resp = client.invoke(prompt)
            return getattr(resp, "content", str(resp))
        except Exception:
            out = client(prompt)
            return getattr(out, "content", str(out))
    except Exception as e:
        return f"(LLM error: {e})"

def synthesize_from_snippets(question: str, docs: List[Dict]) -> str:
    # Filter relevant docs (simple token overlap)
    qtokens = set(re.findall(r"\w+", (question or "").lower()))
    filtered = []
    for d in docs:
        stoks = set(re.findall(r"\w+", (d.get("snippet","")).lower()))
        if qtokens & stoks:
            filtered.append(d)
    if not filtered:
        filtered = docs[:6]  # fallback to top docs

    # Build context
    pieces = []
    for d in filtered[:8]:
        p = f"Source: {d.get('source','')}\nTitle: {d.get('title','')}\nURL: {d.get('url','')}\nContent: {d.get('snippet','')}"
        pieces.append(p)
    context = "\n\n".join(pieces)

    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the context below. "
        "Cite sources inline in square brackets after each sentence, e.g. [Wikipedia]. At the end list sources with titles and URLs.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n"
    )
    return llm_invoke(prompt)

# -----------------------------------------------------------------------------
# Streamlit UI (single run)
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# render chat history
chat_box = st.container()
with chat_box:
    for m in st.session_state.messages:
        cls = "user-msg" if m.get("role") == "user" else "bot-msg"
        st.markdown(f"<div class='{cls}'>{html.escape(m.get('content',''))}</div>", unsafe_allow_html=True)

# input row
cols = st.columns([0.22, 0.10, 0.12, 0.46, 0.06])
with cols[0]:
    uploads = st.file_uploader("Upload files (optional)", type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
with cols[1]:
    audio = mic_recorder(start_prompt="🎙️ Start", stop_prompt="⏹️ Stop", just_once=True, use_container_width=True)
with cols[2]:
    action = st.selectbox("Action", ["Ask", "WebSearch", "Summarize", "Speak"])
with cols[3]:
    user_input = st.text_input("Type your message or paste a YouTube URL...", "")
with cols[4]:
    send = st.button("Send")

# Footer: source / info
st.markdown("---")
st.write("Tip: Set `MCP_URL` in .env to an MCP web-search server to prefer MCP results (free/open MCP servers exist).")

# process audio quickly
if audio:
    st.session_state.messages.append({"role": "user", "content": "🎤 (processing voice...)"} )
    wav_bytes = audio.get("bytes") if isinstance(audio, dict) else audio
    transcript = deepgram_transcribe(wav_bytes) if wav_bytes else ""
    st.session_state.messages[-1] = {"role": "user", "content": transcript or "(couldn't transcribe)"}
    if transcript:
        with st.spinner("Thinking..."):
            reply = llm_invoke(transcript)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            # tts
            tts = deepgram_tts(reply)
            if tts:
                st.audio(tts, format="audio/mpeg")
    safe_rerun()

# when user presses Send
if send and (user_input or uploads):
    # add user message
    qtext = user_input.strip() or "(uploaded files)"
    st.session_state.messages.append({"role": "user", "content": qtext})

    # load uploaded files as docs (RAG)
    docs_context = load_uploaded_files(uploads) if uploads else []

    # check youtube
    vid = extract_youtube_id(user_input)
    if vid:
        yt_trans = fetch_youtube_transcript(vid)
        if yt_trans:
            docs_context.append({"source": "YouTube", "title": f"YouTube {vid}", "snippet": yt_trans[:4000], "url": f"https://youtu.be/{vid}"})
            st.success("YouTube transcript added to context.")

    ai_answer = ""
    with st.spinner("Working..."):
        if action == "Summarize":
            if not docs_context:
                ai_answer = "Please upload files or provide a YouTube URL to summarize."
            else:
                # simple summarization prompt
                context_str = "\n\n".join(f"--- {d['title']} ---\n{d['snippet']}" for d in docs_context)
                prompt = f"Summarize the following documents:\n\n{context_str}\n\nProvide a concise summary."
                ai_answer = llm_invoke(prompt)

        elif action == "WebSearch":
            # 1) run websearch pipeline (MCP first)
            ranked, web_docs = websearch_pipeline(qtext)
            combined_docs = docs_context + web_docs
            # 2) synthesize
            ai_answer = synthesize_from_snippets(qtext, combined_docs)
            # 3) append to messages and show sources below
            st.session_state.messages.append({"role": "assistant", "content": ai_answer})
            st.markdown("**Sources:**")
            for r in ranked[:8]:
                title = r.get("title") or r.get("url") or r.get("source")
                url = r.get("url") or "#"
                st.markdown(f'<span class="source-chip">{html.escape(r.get("source",""))}</span> <a href="{html.escape(url)}" target="_blank">{html.escape(title)}</a>', unsafe_allow_html=True)
            safe_rerun()
            # note: we already appended assistant; return early to avoid double append
        else:  # Ask or Speak
            prompt = qtext
            if docs_context:
                context_str = "\n\n".join(f"--- {d['title']} ---\n{d['snippet']}" for d in docs_context)
                prompt = f"Using ONLY the context below, answer the question.\n\nCONTEXT:\n{context_str}\n\nQUESTION: {qtext}\n\nAnswer:"
            ai_answer = llm_invoke(prompt)
            if action == "Speak":
                tts = deepgram_tts(ai_answer)
                if tts:
                    st.audio(tts, format="audio/mpeg")

    # add assistant message if not already added (WebSearch added earlier)
    if action != "WebSearch":
        st.session_state.messages.append({"role": "assistant", "content": ai_answer})

    safe_rerun()

# Clear chat button
st.markdown("---")
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    safe_rerun()
