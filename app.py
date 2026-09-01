

import os
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


# 1. PAGE CONFIG

st.set_page_config(page_title="AI Document Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI-Powered Document Q&A Chatbot")
# st.caption("RAG based chatbot | Python + LangChain concepts + Gemini API + ChromaDB")


# 2. API KEY SETUP 

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = ""
with st.sidebar:
    st.header("⚙️ Settings")
    if api_key and api_key != "PASTE_YOUR_GEMINI_API_KEY_HERE":
        st.success("API key set ✅")
    else:
        st.error("Paste your API key in the api_key variable at the top of the code")

    st.divider()
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    process_btn = st.button("Process Documents", use_container_width=True)

if api_key:
    genai.configure(api_key=api_key)


# 3. EMBEDDING MODEL (Free, runs locally - no API cost)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

# 4. VECTOR DATABASE SETUP (ChromaDB - free, local)

@st.cache_resource
def get_chroma_collection():
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="documents")
    return collection

collection = get_chroma_collection()


# 5. HELPER FUNCTIONS


def extract_text(file):
    """Extracts raw text from a PDF or TXT file."""
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return file.read().decode("utf-8")


def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into smaller chunks (with overlap for better context)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def add_documents_to_db(files):
    """Processes uploaded files and stores them in ChromaDB."""
    doc_id_counter = 0
    for file in files:
        text = extract_text(file)
        chunks = chunk_text(text)
        embeddings = embedder.encode(chunks).tolist()

        ids = [f"{file.name}_{doc_id_counter + i}" for i in range(len(chunks))]
        metadatas = [{"source": file.name} for _ in chunks]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        doc_id_counter += len(chunks)
    return doc_id_counter


def retrieve_relevant_chunks(query, top_k=3):
    """Retrieves the most relevant chunks related to the user's question."""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0] if results["documents"] else []


def generate_answer(query, context_chunks):
    """Sends context + question to the Gemini LLM to generate an answer."""
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the
context provided below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""

    model = genai.GenerativeModel("gemini-flash-lite-latest")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_text = str(e)
        if "429" in error_text or "ResourceExhausted" in error_text:
            return ("⚠️ Free tier request limit reached for now. "
                    "Please wait for 30-60 seconds and try again.")
        return f"⚠️ An error occurred: {error_text}"



# 6. PROCESS DOCUMENTS BUTTON

if process_btn:
    if not uploaded_files:
        st.sidebar.warning("Please upload some files first")
    else:
        with st.spinner("Processing Documents..."):
            count = add_documents_to_db(uploaded_files)
        st.sidebar.success(f"✅ {count} chunks added to database!")


# 7. CHAT INTERFACE

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask your question...")

if user_query:
    if not api_key:
        st.error("Please add your Gemini API key in the sidebar first")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                relevant_chunks = retrieve_relevant_chunks(user_query)

                if not relevant_chunks:
                    answer = "Please upload and process documents first."
                else:
                    answer = generate_answer(user_query, relevant_chunks)

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})