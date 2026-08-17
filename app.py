import os,re,requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from google.genai import types
from typing import List, Dict, Optional, Tuple
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
from groq import Groq

# ✅ Chat Models (external providers)
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ OCR / Image handling
# import pytesseract
from PIL import Image
# import cv2
from google import genai
from googleapiclient.discovery import build
import io
import base64
from deepgram import DeepgramClient, PrerecordedOptions
import io
import empyrebase
from fpdf import FPDF
from datetime import datetime
import uuid
DG_KEY = os.getenv("voice") 
# Load API keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
SCRAPEDO_API_KEY = os.getenv("SCRAPEDO_API_KEY")
groq_api_key = os.getenv("groq_apikey")
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_client = Groq(api_key=groq_api_key)
# UI
firebaseConfig = {
  'apiKey': "AIzaSyAGflwbj5qTleUs3cerMawam6JhtxcsJKI",
  'authDomain': "chatbot-c38ce.firebaseapp.com",
  'projectId': "chatbot-c38ce",
  'databaseURL':"https://chatbot-c38ce-default-rtdb.firebaseio.com/",
  'storageBucket': "chatbot-c38ce.firebasestorage.app",
  'messagingSenderId': "153975387022",
  'appId': "1:153975387022:web:a4418468e324c40a8daadf",
  'measurementId': "G-L1HSL37CTV"
    
}

firebase = empyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()
def auth_page():
    st.title("Login / Signup")

    choice = st.selectbox("Choose", ["Login", "Signup"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    # ----------------- SIGNUP -----------------
    if choice == "Signup":
        username = st.text_input("Username")

        if st.button("Create Account"):
            try:
                # Create user
                user = auth.create_user_with_email_and_password(email, password)

                # Save username in Firebase DB with localId
                db.child(user['localId']).child("Handle").set(username)

                # Login immediately after signup
                user = auth.sign_in_with_email_and_password(email, password)

                # Store in session
                st.session_state['user'] = user
                st.session_state['username'] = username

                st.success(f"Account created! Welcome {username}")

                st.session_state["page"] = "chatbot"

            except Exception as e:
                st.error(f"Error: {e}")

    # ----------------- LOGIN -----------------
    elif choice == "Login":

        if st.button("Login"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)

                local_id = user['localId']
                username_from_db = db.child(local_id).child("Handle").get().val()

                st.session_state["user"] = user
                st.session_state["username"] = username_from_db
                st.session_state["page"] = "chatbot"
                st.success(f"Welcome {username_from_db}!")
                st.color_picker('click here to go Chatbot page', '#3366ff')
                

                
                
            except Exception as e:
                st.error(f"Login error: {e}")



def _clean(text: str) -> str:

    return re.sub(r"\s+", " ", text).strip()
def extract_youtube_id(url):
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None
def extract_youtube_url(url):
    match= re.search(r"(https?://(?:www\.)?(?:youtube\.com|youtu\.be)/(?:watch\?v=|shorts/|embed/)?[a-zA-Z0-9_-]{11}(?:[^\s]*)?)",url)

    return match.group(1) if match else None
def groq_stt(audio_bytes):
    try:
        response = groq_client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model="whisper-large-v3",
            response_format="json"
        )
        return response.text

    except Exception as e:
        print("Groq STT Error:", e)
        return ""
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
def export_text_to_pdf_bytes(text):
    pdf = FPDF()
    pdf.add_page()

    # Add a Unicode TTF font (DejaVuSans supports most characters)
    pdf.add_font("DejaVu", "", "dejavu-sans/DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.multi_cell(0, 10, text)

    return pdf.output(dest="S").encode("latin-1", "replace")  
# --------- CHATBOT PAGE ---------

def chatbot_page():
    # --- Auth guard ---
    if "user" not in st.session_state or st.session_state["user"] is None:
        st.warning("Please login first!")
        st.session_state["page"] = "auth"
        st.rerun()

    # --- Logout Button ---
    if st.button("Logout"):
        st.session_state.clear()
        st.session_state["page"] = "auth"
        st.rerun()

    # --- Sidebar: Chat History ---
    st.sidebar.title("💬 Chat History")

    user = st.session_state["user"]
    local_id = user.get("localId")
    id_token = user.get("idToken")

    # Debug info (optional)
    st.sidebar.write("debug - local_id:", local_id)
    st.sidebar.write("debug - has idToken:", bool(id_token))

    # --- Load Chats from Firebase ---
    chats = db.child(local_id).child("chats").get().val()
    if chats is None:
        chats = {}

    # Build a sortable list
    items = []
    for cid, c in chats.items():
        timestamp = c.get("timestamp", "")
        items.append((cid, timestamp, c))

    # Sort by timestamp
    items.sort(key=lambda x: x[1])

    # --- Display each chat in sidebar ---
    for chat_id, ts, chat in items:
        with st.sidebar.expander(chat.get("question", "Chat")):
            st.write("**Q:**", chat.get("question"))
            st.write("**A:**", chat.get("answer"))
            st.write("⏱", ts)

            if st.button("Delete Chat", key=f"del_{chat_id}"):
                try:
                    if id_token:
                        db.child(local_id).child("chats").child(chat_id).remove(id_token)
                    else:
                        db.child(local_id).child("chats").child(chat_id).remove()

                    st.success("Deleted!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Delete failed: {e}")

    if st.sidebar.button("🗑 Delete All Chats"):
        try:
            if id_token:
                db.child(local_id).child("chats").remove(id_token)
            else:
                db.child(local_id).child("chats").remove()
            st.sidebar.success("All chats deleted!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Delete all failed: {e}")
               
    st.title("PDF & Image Question Answer Bot")
    choose=st.selectbox("option",["select","Websearch","summarisation","diagram","Best_friend"])
    audio = st.audio_input("🎤 Speak now")    
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
            with st.spinner("Answer via Rag"):
                result = retrieval_chain.invoke({"input": query})
                answer_text=result.get("answer", "").strip()
                st.write(result["answer"])
            
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        chat_data = {
            "question": query,
            "answer": answer_text,
            "timestamp": timestamp,
            "filename": uploaded_file.name
        }

        try:
            if id_token:
                db.child(local_id).child("chats").child(chat_id).set(chat_data, id_token)
            else:
                db.child(local_id).child("chats").child(chat_id).set(chat_data)
            st.success("Saved to Firebase!")
        except Exception as e:
            st.error(f"Failed to save to Firebase: {e}")
            # show raw exception for debugging
            import traceback, sys
            traceback.print_exc()
            pdf_bytes = export_text_to_pdf_bytes(result["answer"])    
        # Show download button
            st.download_button(
                label="📄 Download AI Response as PDF",
                data=pdf_bytes,
                file_name="ai_response.pdf",
                mime="application/pdf"
            )
            st.rerun()
    
                

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
        pdf_bytes = export_text_to_pdf_bytes(response.content)

    # Show download button
        st.download_button(
            label="📄 Download AI Response as PDF",
            data=pdf_bytes,
            file_name="ai_response.pdf",
            mime="application/pdf"
        )

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
                pdf_bytes = export_text_to_pdf_bytes(response.text)

    # Show download button
            st.download_button(
                label="📄 Download AI Response as PDF",
                data=pdf_bytes,
                file_name="ai_response.pdf",
                mime="application/pdf"
            )
        except:        
            st.info("fetch via scrap")
            
            youtube_id = extract_youtube_id(query)
            if youtube_id:
                st.info("fetching youtube details")
            
                yt_text = fetch_youtube_transcript(youtube_id)
                if yt_text.startswith("Error"):
                    st.error(yt_text)
                else:
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
                    pdf_bytes = export_text_to_pdf_bytes(yt_text)

                # Show download button
                    st.download_button(
                        label="📄 Download AI Response as PDF",
                        data=pdf_bytes,
                        file_name="ai_response.pdf",
                        mime="application/pdf"
                    )    
            
    elif choose=="diagram" and query:  
        llm = ChatGroq(
                    api_key=groq_api_key, 
                    model="openai/gpt-oss-120b", 
                    temperature=0.1
                )
        prompt_template=ChatPromptTemplate.from_template(
            """You are an expert in writing Mermaid diagram code. 
                    Your task is to generate ONLY the Mermaid code based on the user's request. 
                    Do not include any explanations or markdown tags like ```mermaid or ```.
                    
                    User Request: {input}"""
        )
        chain=prompt_template|llm
        with st.spinner("Generating Mermaid Code"):
            result=chain.invoke({"input":query})
            mermaid_code = result.content.strip()
        with st.spinner("Rendering diagram..."):
                    encoded_code = base64.urlsafe_b64encode(mermaid_code.encode("utf8")).decode("ascii")
                    diagram_url = f"https://mermaid.ink/img/{encoded_code}"
                    response = requests.get(diagram_url)
                    response.raise_for_status()  # Will raise an error for bad status codes
                    
                    img = Image.open(io.BytesIO(response.content))  
                    st.subheader("Generated Diagram:")
                    st.image(img, caption="Diagram generated from Mermaid code.")
                    st.download_button(
                        label="Download Diagram",
                        data=response.content,
                        file_name="generated_diagram.png",
                        mime="image/png"
                    )  

    elif audio:

        llm = ChatGroq(
                    api_key=groq_api_key, 
                    model="openai/gpt-oss-120b", 
                    temperature=0.1
                )
        audio_bytes = audio.read()
        #st.write(audio_bytes)
        with st.spinner("Transcribing your speech..."):
            text = groq_stt(audio_bytes)
            if not text:
                st.error("Transcription failed.")
                st.stop()
            st.write("### 🗣 You said:")
            st.success(text)   
            with st.spinner("Thinking..."):
                reply = llm.invoke(text).content

        st.write("### 🤖 AI Response:")
        st.write(reply)
        # Create PDF bytes
        pdf_bytes = export_text_to_pdf_bytes(reply)

        # Show download button
        st.download_button(
            label="📄 Download AI Response as PDF",
            data=pdf_bytes,
            file_name="ai_response.pdf",
            mime="application/pdf"
        )

        





    elif choose=="Best_friend" and query:   
        from langchain.chains import LLMChain
        def extract_memory_from_message(query):
            prompt =ChatPromptTemplate.from_template( """
            Extract ONLY important user information from this message.

            Examples:
            - "I am Ram" → User name is Ram.
            - "I like cricket" → User likes cricket.
            - "I live in Kolkata" → User lives in Kolkata.

            If the message contains no personal info, return "none".

            Message: {query}
            """)
            llm = ChatGroq(
                    model="openai/gpt-oss-120b",
                    api_key=groq_api_key
                )
            chain = LLMChain(llm=llm, prompt=prompt)
            memory = chain.run(query=query)
            
            return memory.strip()
        def save_memory(local_id, memory):
            if memory.lower() == "none":
                return
            
            db.child(local_id).child("memory").push(memory)


        def load_user_memory(local_id):
            memories = db.child(local_id).child("memory").get().val()
            if memories is None:
                return ""
            
            combined = "\n".join(memories.values())
            return combined
        def friend_mode_reply(local_id, query):
    # 1. Extract memory (e.g., "User name is Ram")
            new_memory = extract_memory_from_message(query)

            # 2. Save memory
            save_memory(local_id, new_memory)

            # 3. Load ALL past memories
            full_memory = load_user_memory(local_id)

            # 4. Build friendly prompt
            prompt =ChatPromptTemplate.from_template( f"""
            You are the user's friendly AI companion.

            Here are things you KNOW about the user:
            {full_memory}

            User message: {query}

            Reply like a close friend who knows them personally.
            If you know their name, use it naturally.
            """)

            llm = ChatGroq(
                    model="openai/gpt-oss-120b",
                    api_key=groq_api_key
                )
            chain = LLMChain(llm=llm, prompt=prompt)
            reply= chain.run({"question": query})
            return reply
        answer=friend_mode_reply(local_id, query)
        st.write("AI:", answer)

        # Save chat normally
        chat_id = str(uuid.uuid4())
        chat_data = {
            "question": query,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        db.child(local_id).child("chats").child(chat_id).set(chat_data)

    

  
            
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
                pdf_bytes = export_text_to_pdf_bytes(response.text)

            # Show download button
                st.download_button(
                    label="📄 Download AI Response as PDF",
                    data=pdf_bytes,
                    file_name="ai_response.pdf",
                    mime="application/pdf"
                )
                    
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
                pdf_bytes = export_text_to_pdf_bytes(response.content)

            # Show download button
                st.download_button(
                    label="📄 Download AI Response as PDF",
                    data=pdf_bytes,
                    file_name="ai_response.pdf",
                    mime="application/pdf"
                )
             
            
if "page" not in st.session_state:
    
    st.session_state["page"] = "auth"
    
if st.session_state["page"] == "auth":
    auth_page()
        
        
else:
    chatbot_page()

