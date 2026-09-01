# AI-Powered-Document-Q-A-Chatbot
# AI-Powered-Document-Q-A-Chatbot
AI-Powered Document Q&A Chatbot (RAG)

A chatbot that answers questions from your own PDF/TXT documents using Retrieval-Augmented Generation (RAG).
Tech Stack
Python, Streamlit (UI)
Google Gemini API (LLM)
Sentence-Transformers (embeddings)
ChromaDB (vector database)
pypdf (PDF text extraction)
How It Works
Upload PDF/TXT documents
Documents are split into chunks and converted into embeddings
Embeddings are stored in a vector database (ChromaDB)
On each question, the most relevant chunks are retrieved
Retrieved chunks + question are sent to Gemini to generate an accurate answer# AI-Powered-Document-Q-A-Chatbot
