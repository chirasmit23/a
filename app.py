# app.py
# UltraChat — WebSearch + Synthesizer (single-file, improved ranking & semantics)
# Save as app.py and run: streamlit run app.py

import os
import io
import re
import time
import html
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import docx2txt

# NOTE: Removed server-side sounddevice/soundfile usage because cloud servers (Render, Heroku, etc.)
# don't have a microphone device. Recording must happen in the browser (client) — below we embed a small
# client-side recorder using an HTML/JS snippet and ask users to upload the recorded file.
# import sounddevice as sd
# import soundfile as sf

# Optional LLM (Groq). If not installed or not configured, model features are disabled gracefully.
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

# -----------------------
# Config & env
# -----------------------
st.set_page_config(page_title="🎤 UltraChat — WebSearch on Demand", layout="wide")
load_dotenv()

GROQ_KEY = os.getenv("groq_apikey")   # required for LLM synthesis & optional snippet-checking
DG_KEY = os.getenv("voice")           # optional Deepgram for ASR/TTS
STACKEX_KEY = os.getenv("STACKEX_KEY")

TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")
if TESSERACT_PATH and Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

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
    .pulse-dot{width:10px;height:10px;border-radius:50%;background:#7ef0c4;box-shadow:0 0 0 rgba(126,240,196,.6);animation:pulse 1.6s infinite;}
    @keyframes pulse {0%{box-shadow:0 0 0 0 rgba(126,240,196,.6)}70%{box-shadow:0 0 0 16px rgba(126,240,196,0)}100%{box-shadow:0 0 0 0 rgba(126,240,196,0)}}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Utilities
# -----------------------
def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def first_sentence(snippet: str) -> str:
    if not snippet:
        return ""
    s = re.split(r'(?<=[.!?])\s+', snippet.strip())
    return s[0] if s else snippet

def safe_rerun():
    st.rerun()

# -----------------------
# Audio helpers (client-side recording approach)
# -----------------------
# We no longer try to capture audio server-side. Instead we embed a small browser recorder.
# The recorder produces a WAV blob the user can play/download. The user then uploads that file
# via the file uploader. This avoids the "Error querying device -1" problem on Render.

def deepgram_transcribe(wav_bytes: bytes) -> str:
    if not DG_KEY:
        return ""
    try:
        from deepgram import DeepgramClient, PrerecordedOptions, FileSource
        client = DeepgramClient(api_key=DG_KEY)
        src = {"buffer": wav_bytes}
        opts = PrerecordedOptions(model="nova-2", smart_format=True, language="en")
        resp = client.listen.prerecorded.v("1").transcribe_file(src, opts)
        return resp.results.channels[0].alternatives[0].transcript if resp.results.channels else ""
    except Exception:
        return ""

# -----------------------
# Sources (best-effort scrapers / APIs)
# -----------------------
WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
MDN_SEARCH = "https://developer.mozilla.org/api/v1/search?q={}&locale=en-US"

def wikipedia_summary(q: str) -> Optional[Dict]:
    try:
        r = requests.get(WIKI_REST.format(requests.utils.quote(q)), timeout=6)
        if r.status_code != 200:
            return None
        j = r.json()
        extract = _clean(j.get("extract", ""))
        if not extract:
            return None
        return {"source":"Wikipedia","title": j.get("title", q), "snippet": extract, "url": j.get("content_urls", {}).get("desktop", {}).get("page","")}
    except Exception:
        return None

def duckduckgo_instant(q: str, max_results: int = 4) -> List[Dict]:
    try:
        r = requests.get(f"https://api.duckduckgo.com/?q={requests.utils.quote(q)}&format=json&no_html=1", timeout=6)
        j = r.json()
        out = []
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
        if r.status_code != 200:
            return []
        j = r.json()
        out = []
        for doc in j.get("documents", [])[:max_results]:
            title = doc.get("title") or doc.get("slug","")
            summary = _clean(doc.get("summary") or doc.get("excerpt") or "")
            url = doc.get("mdn_url") or f"https://developer.mozilla.org/en-US/docs/{doc.get('slug','')}"
            out.append({"source":"MDN","title":title,"snippet":summary,"url":url})
        return out
    except Exception:
        try:
            guess = f"https://developer.mozilla.org/en-US/docs/Web/API/{q.replace(' ','_')}"
            r2 = requests.get(guess, timeout=4)
            if r2.status_code == 200:
                snippet = _clean(re.sub(r"<[^>]+>", "", r2.text))[:800]
                return [{"source":"MDN","title": q,"snippet": snippet,"url": guess}]
        except Exception:
            pass
        return []

def geeksforgeeks_search(q: str) -> Optional[Dict]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(f"https://www.geeksforgeeks.org/?s={requests.utils.quote(q)}", headers=headers, timeout=6)
        if r.status_code != 200:
            return None
        m = re.search(r'https://www\.geeksforgeeks\.org/[^"\'<> ]+/', r.text)
        if not m:
            guessed = f"https://www.geeksforgeeks.org/{q.replace(' ', '-')}/"
            gm = requests.get(guessed, headers=headers, timeout=4)
            if gm.status_code == 200:
                snippet = _clean(re.sub(r"<[^>]+>", "", gm.text))[:800]
                return {"source":"GeeksforGeeks","title": q,"snippet": snippet,"url": guessed}
            return None
        url = m.group(0)
        page = requests.get(url, headers=headers, timeout=6).text
        snippet = _clean(re.sub(r"<[^>]+>", "", page))[:800]
        return {"source":"GeeksforGeeks","title": url.split("/")[-2].replace("-"," "),"snippet": snippet,"url": url}
    except Exception:
        return None

def stackoverflow_search(q: str, max_items: int = 4) -> List[Dict]:
    try:
        params = {"order":"desc","sort":"relevance","q": q,"site":"stackoverflow","pagesize": max_items}
        if STACKEX_KEY:
            params["key"] = STACKEX_KEY
        r = requests.get("https://api.stackexchange.com/2.3/search/advanced", params=params, timeout=6)
        j = r.json()
        out = []
        for it in j.get("items", [])[:max_items]:
            title = it.get("title","")
            link = it.get("link","")
            snippet = _clean(title)[:600]
            out.append({"source":"StackOverflow","title": title,"snippet": snippet,"url": link})
        return out
    except Exception:
        return []

def arxiv_search(q: str, max_results: int = 2) -> List[Dict]:
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{requests.utils.quote(q)}&start=0&max_results={max_results}"
        r = requests.get(url, timeout=6)
        entries = re.findall(r'<entry>(.*?)</entry>', r.text, flags=re.S)
        out = []
        for e in entries[:max_results]:
            title = re.search(r'<title>(.*?)</title>', e, flags=re.S)
            summary = re.search(r'<summary>(.*?)</summary>', e, flags=re.S)
            link = re.search(r'<id>(.*?)</id>', e, flags=re.S)
            out.append({
                "source":"arXiv",
                "title": (title.group(1).strip() if title else "arXiv"),
                "snippet": _clean((summary.group(1) if summary else "")[:600]),
                "url": link.group(1) if link else ""
            })
        return out
    except Exception:
        return []

# -----------------------
# Ranking & prioritization
# -----------------------
DOMAIN_PRIORITY = {
    "geeksforgeeks.org": 2.2,
    "tutorialspoint.com": 2.0,
    "wikipedia.org": 1.8,
    "developer.mozilla.org": 2.4,
    "stack overflow": 1.2,
    "stackoverflow.com": 1.2,
}

WEB_DEV_HINTS = {"html","css","javascript","js","url","dom","api","http","fetch","react","vue","mdn"}

def looks_webdev(query: str) -> bool:
    q = (query or "").lower()
    return any(tok in q for tok in WEB_DEV_HINTS)

def domain_boost_for_url(url: str) -> float:
    if not url:
        return 0.0
    u = url.lower()
    for dom, boost in DOMAIN_PRIORITY.items():
        if dom in u:
            return boost
    return 0.0

def score_result(res: Dict, query_tokens: List[str]) -> float:
    text = (" ".join([str(res.get("title","")), str(res.get("snippet",""))]) or "").lower()
    overlap = sum(1 for t in query_tokens if t in text)
    src = (res.get("source") or "").lower()
    bias = 0.0
    if src == "wikipedia":
        bias += 0.8
    if src == "stackoverflow":
        bias += 0.3
    if src == "mdn":
        bias += 1.0 if looks_webdev(" ".join(query_tokens)) else -0.7
    bias += domain_boost_for_url(res.get("url",""))
    return overlap + bias

def rank_results(results: List[Dict], query: str) -> List[Dict]:
    toks = re.findall(r"\w+", (query or "").lower())
    return sorted(results, key=lambda r: score_result(r, toks), reverse=True)

# -----------------------
# Aggregator pipeline
# -----------------------
def websearch_pipeline(query: str, use_mdn=True, use_so=True) -> Tuple[List[Dict], List[Dict]]:
    if not query:
        return [], []
    results: List[Dict] = []

    w = wikipedia_summary(query)
    if w: results.append(w)

    results.extend(duckduckgo_instant(query, max_results=4))
    if use_mdn:
        results.extend(mdn_search(query, max_results=3))
    if use_so:
        results.extend(stackoverflow_search(query, max_items=4))

    g = geeksforgeeks_search(query)
    if g:
        results.append(g)

    results.extend(arxiv_search(query, max_results=2))

    # dedupe
    uniq: List[Dict] = []
    seen = set()
    for r in results:
        key = (r.get("url") or r.get("title") or r.get("snippet",""))[:200]
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    ranked = rank_results(uniq, query)
    docs = [{"source": r.get("source"), "title": r.get("title"), "snippet": _clean(r.get("snippet","")), "url": r.get("url")} for r in ranked if r.get("snippet")]
    return ranked, docs

def results_relevant(results: List[Dict], query: str) -> bool:
    if not results:
        return False
    joined = " ".join((r.get("snippet") or "") for r in results).lower()
    toklist = re.findall(r"\w+", (query or "").lower())[:6]
    return len(joined) > 200 or any(tok in joined for tok in toklist)

# -----------------------
# Optional snippet semantic filter (LLM-assisted if available)
# -----------------------
def _is_snippet_relevant_simple(snippet: str, query: str) -> bool:
    if not snippet:
        return False
    q_tokens = set(re.findall(r"\w+", query.lower()))
    s_tokens = set(re.findall(r"\w+", snippet.lower()))
    if len(q_tokens & s_tokens) == 0:
        return False
    if len(snippet) < 40:
        return False
    unrelated = {"distcc","maven","jenkins","docker"}
    if any(u in snippet.lower() for u in unrelated) and not any(u in query.lower() for u in unrelated):
        return False
    return True

def is_snippet_relevant(snippet: str, query: str) -> bool:
    if not _is_snippet_relevant_simple(snippet, query):
        return False
    if ChatGroq and GROQ_KEY:
        try:
            classifier_prompt = (
                f"Yes or no: Does the following snippet directly help answer the question \"{query}\" "
                f"in the context of computer science / technical explanation? Answer 'Yes' or 'No' only.\n\n"
                f"Snippet:\n{snippet[:800]}\n\nAnswer:"
            )
            llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_KEY)
            out = llm.invoke(classifier_prompt)
            text = out.get("output_text") if isinstance(out, dict) else getattr(out, "content", str(out))
            return "yes" in text.lower()
        except Exception:
            return True
    return True

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
        return f"(LLM error: {e})"

def synthesize_from_snippets(question: str, docs: List[Dict]) -> str:
    filtered = []
    for d in docs[:20]:
        snip = d.get("snippet","")
        if is_snippet_relevant(snip, question):
            filtered.append(d)
    if not filtered:
        return "(No sufficiently relevant live snippets found to synthesize an answer.)"

    labeled = []
    for d in filtered[:12]:
        src = d.get("source","Web")
        title = d.get("title") or src
        snippet = d.get("snippet","")[:1600]
        labeled.append(f"[{src}] {title}\n{snippet}")
    context = "\n\n".join(labeled)

    prompt = (
        "You are a careful, factual synthesizer. Use ONLY the snippets below as source material. "
        "Write a concise clear answer to the user's question that combines consensus across sources. "
        "If sources disagree, point out the disagreement and which source seems more authoritative (and why). "
        "Add short inline bracket citations like [Wikipedia], [MDN], [GeeksforGeeks], [StackOverflow] at the end of sentences where you used those sources. "
        "Finally append a short 'Sources' bullet list with the source name and URL for the top sources used.\n\n"
        f"SNIPPETS:\n{context}\n\nQUESTION: {question}\n\nANSWER (concise, source-marked):\n"
    )
    return _llm_invoke(prompt)

# -----------------------
# File upload loaders
# -----------------------
def load_uploaded_files(upload_list) -> List[Dict]:
    out: List[Dict] = []
    if not upload_list:
        return out
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
            elif f.name.lower().endswith((".wav", ".mp3", ".m4a", ".ogg")):
                # for audio: keep raw bytes in snippet so it can be processed by ASR if desired
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

# -----------------------
# Typo correction simple
# -----------------------
COMMON_TYPO = {"complier":"compiler", "compilier":"compiler"}
def normalize_query(q: str) -> str:
    return " ".join(COMMON_TYPO.get(w.lower(), w) for w in q.split())

# -----------------------
# UI
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎤 UltraChat — WebSearch on Demand")
st.caption("Pick 'Ask' for LLM-only answers (or include uploaded docs). Pick 'WebSearch' to fetch MDN / GfG / StackOverflow / Wikipedia / DuckDuckGo / arXiv and synthesize a single source-marked answer.")

# show chat history
for m in st.session_state.messages:
    cls = "user-msg" if m.get("role") == "user" else "bot-msg"
    st.markdown(f"<div class='{cls}'>{m.get('content')}</div>", unsafe_allow_html=True)

cols = st.columns([0.16, 0.08, 0.12, 0.52, 0.06])
with cols[0]:
    uploads = st.file_uploader(
        "Upload files (optional). To transcribe: upload a recorded WAV/MP3 after recording from the browser recorder below.",
        type=["pdf","docx","txt","png","jpg","jpeg","webp","wav","mp3","m4a","ogg"],
        accept_multiple_files=True,
        key="uploader_main",
    )
    st.markdown(
        """
        <small>
        NOTE: Do NOT rely on server-side microphone on cloud hosts. Use the built-in browser recorder (click 🎙️ Record) — it will create a WAV you can download and then upload here. 
        </small>
        """,
        unsafe_allow_html=True,
    )

with cols[1]:
    # Browser-side recorder embedded via HTML/JS.
    # It does NOT send audio to the server automatically (avoids Render error).
    # User should Download the recorded WAV, then Upload via the uploader above for processing.
    if st.button("🎙️ Record (browser)", key="btn_record"):
        # show the recorder UI in a modal-like expander
        with st.expander("Browser Recorder — Record, Play, Download, then Upload the WAV above"):
            recorder_html = """
            <div>
              <p><strong>Browser Recorder</strong> — Click Record, speak, then Stop. You can Play or Download the WAV. After downloading, upload the file via the Streamlit 'Upload files' control.</p>
              <button id="record">Record</button>
              <button id="stop" disabled>Stop</button>
              <button id="play" disabled>Play</button>
              <a id="download" style="display:inline-block;margin-left:10px;"></a>
              <p id="status"></p>
              <audio id="audio" controls style="display:block;margin-top:10px;"></audio>
            </div>
            <script>
            const recordBtn = document.getElementById('record');
            const stopBtn = document.getElementById('stop');
            const playBtn = document.getElementById('play');
            const downloadLink = document.getElementById('download');
            const status = document.getElementById('status');
            const audioEl = document.getElementById('audio');

            let mediaRecorder;
            let audioChunks = [];

            async function init() {
              try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.addEventListener("dataavailable", event => {
                  if (event.data.size > 0) audioChunks.push(event.data);
                });

                mediaRecorder.addEventListener("stop", () => {
                  const blob = new Blob(audioChunks, { type: 'audio/wav' });
                  audioChunks = [];
                  const url = URL.createObjectURL(blob);
                  audioEl.src = url;
                  playBtn.disabled = false;
                  const filename = 'recording_' + Date.now() + '.wav';
                  downloadLink.href = url;
                  downloadLink.download = filename;
                  downloadLink.innerText = 'Download WAV';
                  status.innerText = 'Recording ready. Download and upload via the Streamlit uploader.';
                });
                status.innerText = 'Ready to record (browser mic).';
              } catch (e) {
                status.innerText = 'Microphone access denied or not available: ' + e;
                recordBtn.disabled = true;
              }
            }

            recordBtn.onclick = () => {
              if (!mediaRecorder) return;
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
            st.components.v1.html(recorder_html, height=260)

with cols[2]:
    action = st.selectbox("Action", ["Ask", "WebSearch", "Summarize", "Speak"], index=0, key="action_select")
with cols[3]:
    user_text = st.text_input("Type your message...", key="input_text").strip()
with cols[4]:
    send_clicked = st.button("Send", key="btn_send")

# -----------------------
# Main behavior
# -----------------------
if send_clicked and (user_text or uploads):
    query = normalize_query(user_text or "")
    st.session_state.messages.append({"role":"user","content": query or "(uploaded files only)"})

    uploaded_docs = load_uploaded_files(uploads)

    # If any uploaded file is audio and DG_KEY is configured, transcribe those audio files
    for d in uploaded_docs:
        if d.get("snippet", "").startswith("__AUDIO__"):
            # snippet format used above: "__AUDIO__:{n}_bytes" and url contains path
            # We'll attempt to transcribe by reading bytes from disk if DG_KEY present
            try:
                audio_path = Path(d.get("url"))
                if audio_path.exists() and DG_KEY:
                    wav_bytes = audio_path.read_bytes()
                    transcript = deepgram_transcribe(wav_bytes)
                    if transcript:
                        st.session_state.messages.append({"role":"user","content": transcript})
            except Exception:
                pass

    # Summarize uploaded docs
    if action == "Summarize":
        if not uploaded_docs:
            st.warning("Please upload files to summarize.")
        else:
            ctx = "\n\n".join(f"[{d['source']}] {d['title']}\n{d['snippet'][:1600]}" for d in uploaded_docs[:8])
            prompt = f"Summarize the following documents concisely and clearly:\n\n{ctx}\n\nSummary:"
            summary = _llm_invoke(prompt)
            st.session_state.messages.append({"role":"assistant","content": summary})
            safe_rerun()

    # Ask -> LLM only (no websearch)
    elif action == "Ask":
        if uploaded_docs:
            ctx = "\n\n".join(f"[{d['source']}] {d['title']}\n{d['snippet'][:2000]}" for d in uploaded_docs[:8])
            prompt = f"Use the documents below to answer the question. If they don't contain the answer, answer to the best of your ability.\n\n{ctx}\n\nQ: {query}\nA:"
            ans = _llm_invoke(prompt)
        else:
            ans = _llm_invoke(query)
        st.session_state.messages.append({"role":"assistant","content": ans})
        safe_rerun()

    # WebSearch -> multi-source + synthesize
    elif action == "WebSearch":
        badge = st.empty()
        badge.markdown('<div class="websearch-badge"><div class="pulse-dot"></div> WebSearching…</div>', unsafe_allow_html=True)

        try:
            results, docs = websearch_pipeline(query)
            if uploaded_docs:
                docs = uploaded_docs + docs
            time.sleep(0.2)
        except Exception as e:
            results, docs = [], []
            st.error(f"Search error: {e}")
        finally:
            badge.empty()

        if results:
            top = results[0]
            st.markdown(f"**Top source — {top.get('source','Web')}**: {first_sentence(top.get('snippet',''))}")
            if top.get('title') or top.get('url'):
                st.markdown(f"*{html.escape(top.get('title') or '')} — <{top.get('url','#')}>*")
        else:
            st.info("No live snippets found for that query.")

        if results and results_relevant(results, query):
            answer = synthesize_from_snippets(query, docs)
        else:
            if uploaded_docs:
                ctx = "\n\n".join(f"[{d['source']}] {d['title']}\n{d['snippet'][:2000]}" for d in uploaded_docs[:8])
                prompt = f"Use the documents below to answer the question. If they don't contain the answer, answer to the best of your ability.\n\n{ctx}\n\nQ: {query}\nA:"
                answer = _llm_invoke(prompt)
            else:
                answer = _llm_invoke(query)

        st.session_state.messages.append({"role":"assistant","content": answer})

        st.markdown("**Sources used (top-ranked):**")
        if results:
            for r in results[:12]:
                title = _clean(r.get("title") or "")
                url = r.get("url") or "#"
                st.markdown(f'<span class="source-chip">{html.escape(r.get("source","Web"))}</span> <a href="{url}" target="_blank">{html.escape(title or url)}</a>', unsafe_allow_html=True)
        else:
            st.info("No sources to display.")

        safe_rerun()

    # Speak -> produce answer then TTS
    elif action == "Speak":
        if uploaded_docs:
            ctx = "\n\n".join(f"[{d['source']}] {d['title']}\n{d['snippet'][:2000]}" for d in uploaded_docs[:8])
            prompt = f"Use the documents below to answer the question. If they don't contain the answer, answer to the best of your ability.\n\n{ctx}\n\nQ: {query}\nA:"
            ans = _llm_invoke(prompt)
        else:
            ans = _llm_invoke(query)

        st.session_state.messages.append({"role":"assistant","content": ans})

        if DG_KEY:
            try:
                resp = requests.post("https://api.deepgram.com/v1/speak?model=aura-asteria-en",
                                     headers={"Authorization": f"Token {DG_KEY}"},
                                     json={"text": ans}, timeout=30)
                resp.raise_for_status()
                st.audio(resp.content, format="audio/wav")
            except Exception as e:
                st.error(f"TTS failed: {e}")
        else:
            st.info("Deepgram TTS not configured (set DG_KEY).")
        safe_rerun()
