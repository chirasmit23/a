# app.py
# UltraChat — WebSearch on Demand (single-file) — robust browser recorder -> server -> ASR -> LLM -> TTS
# Save as app.py and run: streamlit run app.py

import os
import io
import re
import time
import base64
import html
import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import docx2txt
import streamlit.components.v1 as components

# Optional LLM (Groq). If not installed or not configured, model features are disabled gracefully.
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

# --- config/env ---
st.set_page_config(page_title="🎤 UltraChat — WebSearch on Demand", layout="wide")
load_dotenv()

GROQ_KEY = os.getenv("groq_apikey")   # LLM (optional)
DG_KEY = os.getenv("voice")           # Deepgram API key (ASR & TTS) (optional)
STACKEX_KEY = os.getenv("STACKEX_KEY")

TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")
if TESSERACT_PATH and Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# logging
logger = logging.getLogger("ultrachat")
logger.setLevel(logging.INFO)

# --- style ---
st.markdown("""
    <style>
    .user-msg { background: rgba(56,139,253,.10); padding:12px; border-radius:10px; margin:6px 0; }
    .bot-msg  { background: rgba(120,120,120,.06); padding:12px; border-radius:10px; margin:6px 0; }
    .source-chip{display:inline-block;margin:4px 6px 0 0;padding:4px 8px;border-radius:999px;border:1px solid #2b3d4f;color:#b7c9d9;background:#0f1b28;font-size:12px}
    .websearch-badge { display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;background:#071422;color:#7ef0c4;font-weight:600;border:1px solid #133040;margin:6px 0 10px 0;}
    .pulse-dot{width:10px;height:10px;border-radius:50%;background:#7ef0c4;box-shadow:0 0 0 rgba(126,240,196,.6);animation:pulse 1.6s infinite;}
    @keyframes pulse {0%{box-shadow:0 0 0 0 rgba(126,240,196,.6)}70%{box-shadow:0 0 0 16px rgba(126,240,196,0)}100%{box-shadow:0 0 0 0 rgba(126,240,196,0)}}
    </style>
""", unsafe_allow_html=True)

# --- small utilities ---
def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def safe_rerun():
    st.rerun()

def first_sentence(snippet: str) -> str:
    if not snippet: return ""
    s = re.split(r'(?<=[.!?])\s+', snippet.strip())
    return s[0] if s else snippet

# --- Deepgram helpers (ASR + TTS) ---
def deepgram_transcribe(audio_bytes: bytes, filename: str = "recording.webm") -> str:
    """Transcribe bytes with Deepgram REST API. Returns transcript or empty string."""
    if not DG_KEY:
        return ""
    try:
        headers = {"Authorization": f"Token {DG_KEY}"}
        files = {"file": (filename, audio_bytes)}
        # use /v1/listen (Deepgram's REST ASR endpoint)
        resp = requests.post("https://api.deepgram.com/v1/listen?punctuate=true", headers=headers, files=files, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        # try to find transcript
        try:
            return j["results"]["channels"][0]["alternatives"][0].get("transcript","") or ""
        except Exception:
            return ""
    except Exception as e:
        logger.exception("Deepgram transcribe error")
        return ""

def deepgram_tts(text: str) -> Optional[bytes]:
    """Request Deepgram TTS (returns audio bytes)"""
    if not DG_KEY or not text:
        return None
    try:
        headers = {"Authorization": f"Token {DG_KEY}", "Content-Type": "application/json"}
        payload = {"text": text}
        resp = requests.post("https://api.deepgram.com/v1/speak?model=aura-asteria-en", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception:
        logger.exception("Deepgram TTS error")
        return None

# --- web search, scraping, LLM synth (kept similar) ---
WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
MDN_SEARCH = "https://developer.mozilla.org/api/v1/search?q={}&locale=en-US"

def wikipedia_summary(q: str) -> Optional[Dict]:
    try:
        r = requests.get(WIKI_REST.format(requests.utils.quote(q)), timeout=6)
        if r.status_code != 200: return None
        j = r.json()
        extract = _clean(j.get("extract",""))
        if not extract: return None
        return {"source":"Wikipedia","title": j.get("title", q), "snippet": extract, "url": j.get("content_urls",{}).get("desktop",{}).get("page","")}
    except Exception:
        return None

def duckduckgo_instant(q: str, max_results: int = 4) -> List[Dict]:
    try:
        r = requests.get(f"https://api.duckduckgo.com/?q={requests.utils.quote(q)}&format=json&no_html=1", timeout=6)
        j = r.json()
        out=[]
        for item in j.get("RelatedTopics", [])[:max_results]:
            if "Text" in item and "FirstURL" in item:
                out.append({"source":"DuckDuckGo","title": item.get("Text","")[:120],"snippet":item.get("Text",""),"url": item.get("FirstURL")})
            else:
                for sub in item.get("Topics", [])[:max_results]:
                    if "Text" in sub and "FirstURL" in sub:
                        out.append({"source":"DuckDuckGo","title": sub.get("Text","")[:120],"snippet": sub.get("Text",""),"url": sub.get("FirstURL")})
        return out
    except Exception:
        return []

def mdn_search(q: str, max_results: int = 3) -> List[Dict]:
    try:
        r = requests.get(MDN_SEARCH.format(requests.utils.quote(q)), timeout=6)
        if r.status_code != 200: return []
        j = r.json()
        out=[]
        for doc in j.get("documents",[])[:max_results]:
            title = doc.get("title") or doc.get("slug","")
            summary = _clean(doc.get("summary") or doc.get("excerpt") or "")
            url = doc.get("mdn_url") or f"https://developer.mozilla.org/en-US/docs/{doc.get('slug','')}"
            out.append({"source":"MDN","title":title,"snippet":summary,"url":url})
        return out
    except Exception:
        return []

# (other search functions like geeksforgeeks_search, stackoverflow_search, arxiv_search can be added as before)
# For brevity, include minimal search functions. If you want the full set, re-use those from your prior file.

# --- simple ranking / scoring (kept simple) ---
def rank_results(results: List[Dict], query: str) -> List[Dict]:
    return results  # keep original ranking simple for brevity

def websearch_pipeline(query: str) -> Tuple[List[Dict], List[Dict]]:
    if not query: return [], []
    results = []
    w = wikipedia_summary(query)
    if w: results.append(w)
    results.extend(duckduckgo_instant(query, max_results=4))
    results.extend(mdn_search(query, max_results=3))
    ranked = rank_results(results, query)
    docs = [{"source": r.get("source"), "title": r.get("title"), "snippet": _clean(r.get("snippet","")), "url": r.get("url")} for r in ranked if r.get("snippet")]
    return ranked, docs

# --- LLM (Groq) invocation (if configured) ---
def _llm_invoke(prompt: str) -> str:
    if ChatGroq is None or not GROQ_KEY:
        return "(LLM not configured — set GROQ_KEY and install langchain_groq for synthesis.)"
    try:
        llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_KEY)
        out = llm.invoke(prompt)
        if isinstance(out, dict):
            return out.get("output_text") or out.get("answer") or str(out)
        return getattr(out, "content", str(out))
    except Exception as e:
        logger.exception("LLM invoke error")
        return f"(LLM error: {e})"

def synthesize_from_snippets(question: str, docs: List[Dict]) -> str:
    if not docs:
        return "(No snippets to synthesize from.)"
    labeled = []
    for d in docs[:8]:
        labeled.append(f"[{d.get('source')}] {d.get('title')}\n{d.get('snippet')[:800]}")
    context = "\n\n".join(labeled)
    prompt = (
        "You are a careful synthesizer. Use only the snippets below to answer the question. "
        "Cite sources inline with [Wikipedia], [MDN], [DuckDuckGo], etc.\n\n"
        f"SNIPPETS:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n"
    )
    return _llm_invoke(prompt)

# --- file loaders (images/docs/audio) ---
def load_uploaded_files(upload_list) -> List[Dict]:
    out=[]
    if not upload_list: return out
    updir = Path("uploads"); updir.mkdir(exist_ok=True)
    for f in upload_list:
        p = updir / f.name
        p.write_bytes(f.getbuffer())
        try:
            if f.name.lower().endswith(".pdf"):
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    docs = PyPDFLoader(str(p)).load()
                    text = "\n\n".join(d.page_content for d in docs)[:3000]
                except Exception:
                    text = ""
            elif f.name.lower().endswith(".docx"):
                text = docx2txt.process(str(p))[:3000]
            elif f.name.lower().endswith(".txt"):
                text = p.read_text(errors="ignore")[:3000]
            elif f.name.lower().endswith((".wav", ".mp3", ".m4a", ".ogg", ".webm")):
                data = p.read_bytes()
                out.append({"source": f.name, "title": f.name, "snippet": f"__AUDIO__:{len(data)}_bytes", "url": str(p)})
                continue
            else:
                text = pytesseract.image_to_string(Image.open(p))[:3000]
            if _clean(text):
                out.append({"source": f.name, "title": f.name, "snippet": _clean(text), "url": ""})
        except Exception:
            continue
    return out

# --- small typo fixer ---
COMMON_TYPO = {"complier":"compiler","compilier":"compiler"}
def normalize_query(q: str) -> str:
    return " ".join(COMMON_TYPO.get(w.lower(), w) for w in q.split())

# --- session init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_b64" not in st.session_state:
    st.session_state.last_audio_b64 = None
if "last_audio_processed" not in st.session_state:
    st.session_state.last_audio_processed = False

# --- UI ---
st.title("🎤 UltraChat — WebSearch on Demand (recorder → server → ASR → LLM → TTS)")
st.caption("Click Record, speak your question (e.g. 'Who is Narendra Modi?'), stop — the audio is sent to the app, transcribed, answered, and optionally spoken back.")

# show chat history
for m in st.session_state.messages:
    cls = "user-msg" if m.get("role")=="user" else "bot-msg"
    st.markdown(f"<div class='{cls}'>{m.get('content')}</div>", unsafe_allow_html=True)

cols = st.columns([0.18, 0.16, 0.12, 0.46, 0.06])
with cols[0]:
    uploads = st.file_uploader("Upload files (optional)", type=["pdf","docx","txt","png","jpg","jpeg","webp","wav","mp3","m4a","ogg","webm"], accept_multiple_files=True, key="uploader_main")
    st.markdown("<small>Recorded audio will be sent automatically to the app for transcription & processing.</small>", unsafe_allow_html=True)

with cols[1]:
    st.markdown("**Recorder**")
    recorder_html = r"""
    <div>
      <div style="font-family:Arial,Helvetica,sans-serif">
        <div style="margin-bottom:6px;font-weight:600">Browser Recorder — click Record, speak, then Stop</div>
        <button id="record">Record</button>
        <button id="stop" disabled>Stop</button>
        <button id="play" disabled>Play</button>
        <span id="status" style="margin-left:8px"></span>
        <audio id="audio" controls style="display:block;margin-top:8px"></audio>
      </div>
    </div>
    <script>
    const recordBtn = document.getElementById('record');
    const stopBtn = document.getElementById('stop');
    const playBtn = document.getElementById('play');
    const status = document.getElementById('status');
    const audioEl = document.getElementById('audio');

    let mediaRecorder;
    let audioChunks = [];

    async function init() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.addEventListener("dataavailable", e => {
          if (e.data && e.data.size > 0) audioChunks.push(e.data);
        });
        mediaRecorder.addEventListener("stop", async () => {
          const blob = new Blob(audioChunks, { type: audioChunks[0]?.type || 'audio/webm' });
          audioChunks = [];
          const url = URL.createObjectURL(blob);
          audioEl.src = url;
          playBtn.disabled = false;
          status.innerText = 'Recording ready — sending to app...';

          const reader = new FileReader();
          reader.onloadend = () => {
            const b64data = reader.result; // data:audio/..;base64,...
            // Preferred: use Streamlit.setComponentValue if available
            try {
              if (window.parent && window.parent.Streamlit && typeof window.parent.Streamlit.setComponentValue === "function") {
                window.parent.Streamlit.setComponentValue(b64data);
              } else if (window.streamlit && typeof window.streamlit.setComponentValue === "function") {
                window.streamlit.setComponentValue(b64data);
              } else {
                // fallback - still postMessage
                window.parent.postMessage({streamlitAudio:true, data:b64data}, "*");
              }
              status.innerText = 'Sent to app.';
            } catch (err) {
              status.innerText = 'Failed to send: ' + err;
            }
          };
          reader.readAsDataURL(blob);
        });
        status.innerText = 'Ready to record (browser mic).';
      } catch (e) {
        status.innerText = 'Microphone access denied or not available: ' + e;
        recordBtn.disabled = true;
      }
    }

    recordBtn.onclick = () => {
      if (!mediaRecorder) return;
      audioChunks = [];
      mediaRecorder.start();
      recordBtn.disabled = true;
      stopBtn.disabled = false;
      status.innerText = 'Recording...';
    };
    stopBtn.onclick = () => {
      if (!mediaRecorder) return;
      mediaRecorder.stop();
      recordBtn.disabled = false;
      stopBtn.disabled = true;
    };
    playBtn.onclick = () => {
      if (audioEl.src) audioEl.play();
    };

    init();
    </script>
    """
    rec_result = components.html(recorder_html, height=240)
    # rec_result might be: None, data-url string, or other; we'll accept whatever and store
    if rec_result:
        # normalize & store as string
        try:
            rec_str = rec_result if isinstance(rec_result, str) else str(rec_result)
            st.session_state.last_audio_b64 = rec_str
            st.session_state.last_audio_processed = False
            st.success("Audio received from browser recorder.")
        except Exception:
            st.error("Received recorder payload but failed to interpret it.")

with cols[2]:
    action = st.selectbox("Action", ["Ask", "WebSearch", "Summarize", "Speak"], index=0, key="action_select")

with cols[3]:
    user_text = st.text_input("Type your message (or leave empty to use recorded input)...", key="input_text").strip()

with cols[4]:
    send_clicked = st.button("Send", key="btn_send")

# --- core pipeline: process query & respond (text + optional TTS) ---
def process_query_and_respond(query: str, uploaded_docs: List[Dict], action: str):
    # append user
    st.session_state.messages.append({"role":"user","content": query})
    # Ask
    if action == "Ask":
        if uploaded_docs:
            ctx = "\n\n".join(f"[{d['source']}] {d['title']}\n{d['snippet'][:2000]}" for d in uploaded_docs[:8])
            prompt = f"Use the documents below to answer the question.\n\n{ctx}\n\nQ: {query}\nA:"
            ans = _llm_invoke(prompt)
        else:
            ans = _llm_invoke(query)
        st.session_state.messages.append({"role":"assistant","content": ans})
        return ans
    # WebSearch
    if action == "WebSearch":
        results, docs = websearch_pipeline(query)
        if uploaded_docs:
            docs = uploaded_docs + docs
        if results and len(results)>0:
            # synthesize
            ans = synthesize_from_snippets(query, docs)
        else:
            ans = _llm_invoke(query)
        st.session_state.messages.append({"role":"assistant","content": ans})
        return ans
    # Summarize
    if action == "Summarize":
        if not uploaded_docs:
            st.warning("Upload files to summarize.")
            return ""
        ctx = "\n\n".join(f"[{d['source']}] {d['title']}\n{d['snippet'][:1600]}" for d in uploaded_docs[:8])
        prompt = f"Summarize the following documents concisely and clearly:\n\n{ctx}\n\nSummary:"
        summary = _llm_invoke(prompt)
        st.session_state.messages.append({"role":"assistant","content": summary})
        return summary
    # Speak
    if action == "Speak":
        if uploaded_docs:
            ctx = "\n\n".join(f"[{d['source']}] {d['title']}\n{d['snippet'][:2000]}" for d in uploaded_docs[:8])
            prompt = f"Use the documents below to answer the question.\n\n{ctx}\n\nQ: {query}\nA:"
            ans = _llm_invoke(prompt)
        else:
            ans = _llm_invoke(query)
        st.session_state.messages.append({"role":"assistant","content": ans})
        # TTS (Deepgram)
        if DG_KEY:
            tts_bytes = deepgram_tts(ans)
            if tts_bytes:
                st.audio(tts_bytes, format="audio/wav")
            else:
                st.error("TTS failed.")
        else:
            st.info("Deepgram TTS not configured (set DG_KEY).")
        return ans

# --- handle manual Send ---
if send_clicked and (user_text or uploads):
    q = normalize_query(user_text or "")
    uploaded_docs = load_uploaded_files(uploads)
    process_query_and_respond(q, uploaded_docs, action)
    safe_rerun()

# --- robust handler for recorded data URL ---
def handle_recording_b64(data_url: str):
    """Tolerant processing of recorder output. Accept strings, handle permission errors, decode base64, transcribe, process."""
    if not data_url:
        return

    # convert non-strings into string
    if not isinstance(data_url, str):
        try:
            data_url = str(data_url)
        except Exception:
            st.error("Recorder returned a non-string payload we could not parse.")
            return

    # check for common permission errors or status messages
    low = data_url.lower()
    if "notallowed" in low or "permission" in low or "denied" in low:
        st.warning("Microphone access denied in your browser. Please allow microphone access and reload the page.")
        st.session_state.last_audio_processed = True
        return

    # ensure it's a data URL with comma separator
    if "," not in data_url or not data_url.startswith("data:"):
        st.error("Recorder returned unexpected payload (not a data: URL). Try recording again. If this persists, use the browser recorder download/upload fallback.")
        st.session_state.last_audio_processed = True
        return

    header, b64 = data_url.split(",", 1)
    m = re.match(r"data:(audio/[^;]+);base64", header)
    mime = m.group(1) if m else "audio/webm"
    ext = "webm"
    if "wav" in mime:
        ext = "wav"
    elif "ogg" in mime:
        ext = "ogg"
    elif "mpeg" in mime or "mp3" in mime:
        ext = "mp3"

    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        st.error(f"Failed to decode recorded audio: {e}")
        st.session_state.last_audio_processed = True
        return

    updir = Path("uploads"); updir.mkdir(exist_ok=True)
    fname = updir / f"recording_{int(time.time()*1000)}.{ext}"
    try:
        fname.write_bytes(raw)
        st.info(f"Saved recording to {fname}")
    except Exception as e:
        st.error(f"Failed to save recording: {e}")
        st.session_state.last_audio_processed = True
        return

    # Transcribe automatically if DG_KEY present
    transcript = ""
    if DG_KEY:
        with st.spinner("Transcribing audio..."):
            transcript = deepgram_transcribe(raw, filename=fname.name)
    else:
        st.info("DG_KEY not configured — skipping ASR. You can upload the saved file to the uploader to process it manually later.")

    if transcript:
        st.success("Transcription available.")
        # Create a minimal uploaded_docs entry so the pipeline knows there's an audio file
        uploaded_docs = [{"source": fname.name, "title": fname.name, "snippet": f"__AUDIO__:{len(raw)}_bytes", "url": str(fname)}]
        # Use the transcript as the query and run pipeline automatically
        process_query_and_respond(transcript, uploaded_docs, action)
        st.session_state.last_audio_processed = True
        # If Speak action, process_query_and_respond will already attempt TTS
        safe_rerun()
    else:
        st.warning("No transcript produced (Deepgram not configured or ASR failed). Upload the saved file via the uploader to process manually.")
        st.session_state.last_audio_processed = True

# --- if recorder produced something and not yet processed, handle it ---
if st.session_state.get("last_audio_b64") and not st.session_state.get("last_audio_processed", False):
    try:
        handle_recording_b64(st.session_state.get("last_audio_b64"))
    except Exception as e:
        logger.exception("Error handling recorded audio")
        st.error(f"Error processing recorded audio: {e}")
        st.session_state.last_audio_processed = True
    finally:
        # clear stored base64 to avoid re-processing repeatedly; keep processed flag to prevent loops
        st.session_state.last_audio_b64 = None

# End of file
