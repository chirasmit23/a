# app.py
# UltraChat — Voice RAG (attach image → record → ASR → LLM → TTS)
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

# -----------------------
# Config & env
# -----------------------
st.set_page_config(page_title="🎤 UltraChat Voice RAG", layout="wide")
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

# -----------------------
# Styling
# -----------------------
st.markdown(
    """
    <style>
    .user-msg { background: rgba(56,139,253,.10); padding:12px; border-radius:10px; margin:6px 0; }
    .bot-msg  { background: rgba(120,120,120,.06); padding:12px; border-radius:10px; margin:6px 0; }
    .source-chip{display:inline-block;margin:4px 6px 0 0;padding:4px 8px;border-radius:999px;border:1px solid #2b3d4f;color:#b7c9d9;background:#0f1b28;font-size:12px}
    .websearch-badge { display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;background:#071422;color:#7ef0c4;font-weight:600;border:1px solid #133040;margin:6px 0 10px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Utilities
# -----------------------
def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def safe_rerun():
    st.experimental_rerun()

def first_sentence(snippet: str) -> str:
    if not snippet:
        return ""
    s = re.split(r'(?<=[.!?])\s+', snippet.strip())
    return s[0] if s else snippet

# -----------------------
# Deepgram helpers (ASR + TTS)
# -----------------------
def deepgram_transcribe(audio_bytes: bytes, filename: str = "recording.wav") -> str:
    """Transcribe bytes with Deepgram REST API (returns transcript or empty string)."""
    if not DG_KEY:
        return ""
    try:
        headers = {"Authorization": f"Token {DG_KEY}"}
        files = {"file": (filename, audio_bytes)}
        resp = requests.post("https://api.deepgram.com/v1/listen?punctuate=true", headers=headers, files=files, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        try:
            return j["results"]["channels"][0]["alternatives"][0].get("transcript","") or ""
        except Exception:
            return ""
    except Exception:
        logger.exception("Deepgram transcribe error")
        return ""

def deepgram_tts(text: str) -> Optional[bytes]:
    """Request Deepgram TTS (returns audio bytes)."""
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

# -----------------------
# Minimal websearch/sources (kept small for clarity; reuse your full functions if desired)
# -----------------------
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

def websearch_pipeline(query: str) -> Tuple[List[Dict], List[Dict]]:
    if not query: return [], []
    results=[]
    w = wikipedia_summary(query)
    if w: results.append(w)
    results.extend(duckduckgo_instant(query, max_results=4))
    results.extend(mdn_search(query, max_results=3))
    docs = [{"source": r.get("source"), "title": r.get("title"), "snippet": _clean(r.get("snippet","")), "url": r.get("url")} for r in results if r.get("snippet")]
    return results, docs

# -----------------------
# LLM invocation & synthesizer
# -----------------------
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
        "You are a careful, factual synthesizer. Use ONLY the snippets below as source material. "
        "Write a concise clear answer and add short bracket citations like [Wikipedia].\n\n"
        f"SNIPPETS:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n"
    )
    return _llm_invoke(prompt)

# -----------------------
# File upload loaders (images/docs/audio)
# -----------------------
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
                # images: OCR
                text = pytesseract.image_to_string(Image.open(p))[:3000]
            if _clean(text):
                out.append({"source": f.name, "title": f.name, "snippet": _clean(text), "url": str(p) if f.name.lower().endswith((".wav", ".mp3", ".m4a", ".ogg", ".webm")) else ""})
        except Exception:
            continue
    return out

# -----------------------
# Typo correction simple
# -----------------------
COMMON_TYPO = {"complier":"compiler", "compilier":"compiler"}
def normalize_query(q: str) -> str:
    return " ".join(COMMON_TYPO.get(w.lower(), w) for w in q.split())

# -----------------------
# UI & Recorder (WAV encoder in JS)
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_b64" not in st.session_state:
    st.session_state.last_audio_b64 = None
if "last_audio_processed" not in st.session_state:
    st.session_state.last_audio_processed = False

st.title("🎤 UltraChat Voice RAG")
st.caption("Attach a photo (optional), click the mic, speak your question, stop — the app will transcribe and answer (and optionally speak back).")

# left column: upload image; center: recorder + action + input
cols = st.columns([0.18, 0.08, 0.12, 0.52, 0.06])

with cols[0]:
    st.markdown("**Attach photo (optional)** — will be OCR'd and used as context.")
    image_file = st.file_uploader("", type=["png","jpg","jpeg","webp","tiff"], key="image_attach", accept_multiple_files=False)
    img_ocr_text = ""
    if image_file:
        # save and OCR
        updir = Path("uploads"); updir.mkdir(exist_ok=True)
        img_path = updir / f"image_{int(time.time()*1000)}_{image_file.name}"
        img_path.write_bytes(image_file.getbuffer())
        try:
            img = Image.open(img_path)
            img_ocr_text = pytesseract.image_to_string(img)[:3000]
            if img_ocr_text.strip():
                st.success("Image attached and OCR text extracted.")
                st.markdown(f"> {img_ocr_text[:200].replace(chr(10),' ')}{'...' if len(img_ocr_text)>200 else ''}")
            else:
                st.info("Image attached but no readable text found by OCR.")
        except Exception as e:
            st.error(f"Image processing failed: {e}")

with cols[1]:
    st.markdown("**Recorder**")
    # Recorder JS: uses AudioContext + ScriptProcessor to encode WAV (16-bit PCM) client-side.
    recorder_html = r"""
    <div>
      <div style="font-family:Arial,Helvetica,sans-serif">
        <div style="font-weight:600;margin-bottom:6px;">Browser Recorder — click Record, speak, then Stop</div>
        <button id="record">Record</button>
        <button id="stop" disabled>Stop</button>
        <span id="status" style="margin-left:8px"></span>
      </div>
    </div>
    <script>
    // WAV encoder using ScriptProcessorNode for broad compatibility
    const recordBtn = document.getElementById('record');
    const stopBtn = document.getElementById('stop');
    const status = document.getElementById('status');

    let audioContext;
    let input;
    let processor;
    let microphoneStream;
    let leftBuf = [];
    let rightBuf = [];
    let recordingLength = 0;
    let sampleRate = 48000;

    function encodeWAV(samples, sampleRate) {
      const buffer = new ArrayBuffer(44 + samples.length * 2);
      const view = new DataView(buffer);
      function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i));
        }
      }
      let offset = 0;
      writeString(view, offset, 'RIFF'); offset += 4;
      view.setUint32(offset, 36 + samples.length * 2, true); offset += 4;
      writeString(view, offset, 'WAVE'); offset += 4;
      writeString(view, offset, 'fmt '); offset += 4;
      view.setUint32(offset, 16, true); offset += 4;
      view.setUint16(offset, 1, true); offset += 2;
      view.setUint16(offset, 1, true); offset += 2;
      view.setUint32(offset, sampleRate, true); offset += 4;
      view.setUint32(offset, sampleRate * 2, true); offset += 4;
      view.setUint16(offset, 2, true); offset += 2;
      view.setUint16(offset, 16, true); offset += 2;
      writeString(view, offset, 'data'); offset += 4;
      view.setUint32(offset, samples.length * 2, true); offset += 4;
      // write PCM samples
      let index = 0;
      const volume = 1;
      for (let i = 0; i < samples.length; i++, index += 1) {
        let s = Math.max(-1, Math.min(1, samples[i] * volume));
        view.setInt16(44 + i*2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }
      return new Blob([view], {type: 'audio/wav'});
    }

    async function startRecording() {
      leftBuf = [];
      rightBuf = [];
      recordingLength = 0;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({audio: true});
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        sampleRate = audioContext.sampleRate || 48000;
        microphoneStream = stream;
        input = audioContext.createMediaStreamSource(stream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = function(e) {
          const left = e.inputBuffer.getChannelData(0);
          leftBuf.push(new Float32Array(left));
          recordingLength += left.length;
        };
        input.connect(processor);
        processor.connect(audioContext.destination);
        status.innerText = 'Recording...';
        recordBtn.disabled = true;
        stopBtn.disabled = false;
      } catch (err) {
        status.innerText = 'Microphone unavailable or permission denied: ' + err;
      }
    }

    function stopRecordingAndSend() {
      // merge buffers
      const samples = new Float32Array(recordingLength);
      let offset = 0;
      for (let i = 0; i < leftBuf.length; i++) {
        samples.set(leftBuf[i], offset);
        offset += leftBuf[i].length;
      }
      // create WAV blob
      const wavBlob = encodeWAV(samples, sampleRate);
      // convert blob to base64 data URL
      const reader = new FileReader();
      reader.onloadend = function() {
        const dataUrl = reader.result; // like data:audio/wav;base64,...
        // send back to Streamlit via the setComponentValue bridge (preferred)
        try {
          if (window.parent && window.parent.Streamlit && typeof window.parent.Streamlit.setComponentValue === "function") {
            window.parent.Streamlit.setComponentValue(dataUrl);
          } else if (window.streamlit && typeof window.streamlit.setComponentValue === "function") {
            window.streamlit.setComponentValue(dataUrl);
          } else {
            // fallback: set to localStorage and reload (rare), but try postMessage
            window.parent.postMessage({streamlitAudio: true, data: dataUrl}, "*");
          }
          status.innerText = 'Recording sent to app.';
        } catch (err) {
          status.innerText = 'Failed to send recording: ' + err;
        }
      };
      reader.readAsDataURL(wavBlob);

      // cleanup
      try {
        if (processor) { processor.disconnect(); processor.onaudioprocess = null; }
        if (input) { input.disconnect(); }
        if (microphoneStream) {
          microphoneStream.getTracks().forEach(t => t.stop());
        }
        if (audioContext && audioContext.close) audioContext.close();
      } catch (e) {}
      recordBtn.disabled = false;
      stopBtn.disabled = true;
    }

    recordBtn.onclick = startRecording;
    stopBtn.onclick = stopRecordingAndSend;
    </script>
    """
    rec_result = components.html(recorder_html, height=220)

    # components.html returns the value passed via Streamlit.setComponentValue inside the iframe.
    if rec_result:
        # accept strings only
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
    user_text = st.text_input("Type your message (or leave empty to use recorded input)...", key="input_text")

with cols[4]:
    send_clicked = st.button("Send", key="btn_send")

# display chat history
for m in st.session_state.messages:
    cls = "user-msg" if m.get("role") == "user" else "bot-msg"
    st.markdown(f"<div class='{cls}'>{m.get('content')}</div>", unsafe_allow_html=True)

# -----------------------
# Pipeline: process query & respond (text + optional TTS)
# -----------------------
def process_query_and_respond(query: str, context_docs: List[Dict], action: str):
    """Given text query and optional context docs, append messages, call LLM / websearch, and optionally TTS."""
    st.session_state.messages.append({"role":"user","content": query})

    if action == "Summarize":
        if not context_docs:
            st.warning("No documents to summarize.")
            return
        ctx = "\n\n".join(f"[{d['source']}] {d['title']}\n{d['snippet'][:1600]}" for d in context_docs[:8])
        prompt = f"Summarize the following documents concisely:\n\n{ctx}\n\nSummary:"
        ans = _llm_invoke(prompt)
        st.session_state.messages.append({"role":"assistant","content": ans})
        return ans

    if action == "Ask":
        if context_docs:
            ctx = "\n\n".join(f"[{d['source']}] {d['title']}\n{d['snippet'][:2000]}" for d in context_docs[:8])
            prompt = f"Use the documents below as context. Answer concisely.\n\n{ctx}\n\nQ: {query}\nA:"
            ans = _llm_invoke(prompt)
        else:
            ans = _llm_invoke(query)
        st.session_state.messages.append({"role":"assistant","content": ans})
        return ans

    if action == "WebSearch":
        results, docs = websearch_pipeline(query)
        if context_docs:
            docs = context_docs + docs
        if results:
            ans = synthesize_from_snippets(query, docs)
        else:
            ans = _llm_invoke(query)
        st.session_state.messages.append({"role":"assistant","content": ans})
        # show sources
        if results:
            st.markdown("**Sources used (top-ranked):**")
            for r in results[:6]:
                st.markdown(f"- {r.get('source')} — {r.get('title') or r.get('url')}")
        return ans

    if action == "Speak":
        if context_docs:
            ctx = "\n\n".join(f"[{d['source']}] {d['title']}\n{d['snippet'][:2000]}" for d in context_docs[:8])
            prompt = f"Use the documents below as context. Answer concisely.\n\n{ctx}\n\nQ: {query}\nA:"
            ans = _llm_invoke(prompt)
        else:
            ans = _llm_invoke(query)
        st.session_state.messages.append({"role":"assistant","content": ans})
        # TTS
        if DG_KEY:
            tts_bytes = deepgram_tts(ans)
            if tts_bytes:
                st.audio(tts_bytes, format="audio/wav")
            else:
                st.error("TTS failed.")
        else:
            st.info("Deepgram TTS not configured (set DG_KEY).")
        return ans

# -----------------------
# Manual send
# -----------------------
if send_clicked and (user_text or image_file):
    q = normalize_query(user_text or "")
    # include image OCR as a doc if present
    context_docs = []
    if image_file and img_ocr_text:
        context_docs.append({"source":"Image(OCR)", "title": image_file.name, "snippet": img_ocr_text, "url": ""})
    process_query_and_respond(q, context_docs, action)
    safe_rerun()

# -----------------------
# Handler for recorded audio arriving from the JS recorder
# -----------------------
def handle_recording_b64(data_url: str):
    """Process data:audio/wav;base64,... payload from the recorder."""
    if not data_url:
        return

    if not isinstance(data_url, str):
        try:
            data_url = str(data_url)
        except Exception:
            st.error("Recorder returned non-string payload.")
            st.session_state.last_audio_processed = True
            return

    low = data_url.lower()
    if "permission" in low or "denied" in low or "notallowed" in low:
        st.warning("Microphone access denied. Allow microphone in your browser and try again.")
        st.session_state.last_audio_processed = True
        return

    if not data_url.startswith("data:audio/") or "," not in data_url:
        st.error("Recorder returned unexpected payload (not a data: audio URL). Try recording again.")
        st.session_state.last_audio_processed = True
        return

    header, b64 = data_url.split(",", 1)
    m = re.match(r"data:(audio/[^;]+);base64", header)
    mime = m.group(1) if m else "audio/wav"
    ext = "wav"
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

    # transcribe with Deepgram if available
    transcript = ""
    if DG_KEY:
        with st.spinner("Transcribing..."):
            transcript = deepgram_transcribe(raw, filename=fname.name)
    else:
        st.info("DG_KEY not configured — skipping automatic transcription. Upload the WAV manually if you want ASR.")

    if transcript:
        st.success("Transcription ready:")
        st.markdown(f"> {transcript}")
        # build context from image OCR if present
        context_docs = []
        if image_file and img_ocr_text:
            context_docs.append({"source":"Image(OCR)", "title": image_file.name, "snippet": img_ocr_text, "url": ""})
        # automatically run the pipeline with the transcript as the query
        process_query_and_respond(transcript, context_docs, action)
        st.session_state.last_audio_processed = True
        safe_rerun()
    else:
        st.warning("No transcript produced. If DG_KEY is not set you can upload the saved WAV via the uploader and press Send manually.")
        st.session_state.last_audio_processed = True

# If recorder produced something and not yet processed, handle it
if st.session_state.get("last_audio_b64") and not st.session_state.get("last_audio_processed", False):
    try:
        handle_recording_b64(st.session_state.get("last_audio_b64"))
    except Exception as e:
        logger.exception("Error processing recorded audio")
        st.error(f"Error processing recorded audio: {e}")
        st.session_state.last_audio_processed = True
    finally:
        st.session_state.last_audio_b64 = None

# -----------------------
# End
# -----------------------
