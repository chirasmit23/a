import os
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
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ✅ Chat Models (external providers)
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ OCR / Image handling
# import pytesseract
# from PIL import Image
# import cv2
from google import genai


# Load API keys
load_dotenv()
groq_api_key = os.getenv("groq_apikey")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# UI
st.title("PDF & Image Question Answer Bot")
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



    if docs and uploaded_file.name.lower().endswith(".pdf"):
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

        if query:
            result = retrieval_chain.invoke({"input": query})
            st.subheader("Answer:")
            st.write(result.get("answer", "No answer could be generated."))
    elif uploaded_file:
        st.info("Could not extract any text from the uploaded file.")
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
    classifier_chain=classifier_llm|classifier_promt
    classification_result = classifier_chain.invoke( query)
    

    from langchain.chains import LLMChain
  
   
    if "realtime" in classification_result:
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

            
